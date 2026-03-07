# D3Q19 MRT Lattice Boltzmann solver for single-phase incompressible flow.
import time

import numpy as np
from loguru import logger

from ._lattice import D3Q19_M, D3Q19_M_inv
from ._taichi_helpers import ensure_taichi


# Pressure BCs: only the difference matters for Darcy's law
_RHO_IN = 1.00
_RHO_OUT = 0.99

# Map axis int to (inlet_face, outlet_face) pairs
_AXIS_FACES = {
    0: ("x0", "x1"),
    1: ("y0", "y1"),
    2: ("z0", "z1"),
}


def _build_solver(solid, axis, nu, sparse):
    """Construct and initialize a D3Q19 MRT flow solver.

    Parameters
    ----------
    solid : np.ndarray
        Binary solid mask (1=solid, 0=pore).
    axis : int
        Flow axis (0=x, 1=y, 2=z).
    nu : float
        Kinematic viscosity in lattice units.
    sparse : bool
        Use Taichi sparse storage.

    Returns
    -------
    solver : _D3Q19Solver
        Initialized solver ready to step.
    """
    ti = ensure_taichi()
    solver = _D3Q19Solver(ti, solid, nu=nu, sparse=sparse)
    inlet, outlet = _AXIS_FACES[axis]
    solver.set_bc_rho(inlet, _RHO_IN)
    solver.set_bc_rho(outlet, _RHO_OUT)
    solver.init_fields()
    return solver


def solve_flow(
    solid, axis, nu=1 / 6, n_steps=100_000, tol=1e-3, log_every=500, sparse=False
):
    """Run LBM flow to steady state and return arrays.

    Parameters
    ----------
    solid : np.ndarray, shape (nx, ny, nz), dtype int8
        Binary solid mask (1=solid, 0=pore).
    axis : int
        Flow axis (0=x, 1=y, 2=z).
    nu : float
        Kinematic viscosity in lattice units. Default 1/6.
        Relaxation time: tau_f = 3*nu + 0.5.
    n_steps : int
        Maximum LBM iterations.
    tol : float or None
        Convergence tolerance on delta|v|/|v|. None disables
        early stopping.
    log_every : int
        Log convergence progress every this many steps.
    sparse : bool
        Use Taichi pointer-backed sparse storage.

    Returns
    -------
    velocity : np.ndarray, shape (nx, ny, nz, 3)
        Steady-state velocity field.
    density : np.ndarray, shape (nx, ny, nz)
        Steady-state density field.
    """
    solver = _build_solver(solid, axis, nu, sparse)
    _run_to_convergence(solver, n_steps, tol, log_every)
    velocity = solver.get_velocity()
    density = solver.get_density()
    return velocity, density


def _run_to_convergence(solver, n_steps, tol, log_every):
    """Step the solver until convergence or max steps."""
    t_start = time.time()
    v_prev = None
    for step in range(n_steps + 1):
        solver.step()
        if step % log_every != 0:
            continue
        elapsed = time.time() - t_start
        v_now = solver.get_velocity()
        if v_prev is not None:
            v_total = np.sum(np.abs(v_now))
            v_change = np.sum(np.abs(v_now - v_prev))
            ratio = v_change / v_total if v_total > 0 else 0.0
            logger.info(
                f"Step {step:>6d}/{n_steps}  "
                f"|v|={v_total:.4e}  "
                f"delta={ratio:.2e}  "
                f"elapsed={elapsed:.1f}s"
            )
            if tol is not None and v_total > 0 and ratio < tol:
                logger.info(
                    f"Converged at step {step} (delta|v|/|v|={ratio:.2e} < tol={tol:.2e})"
                )
                return
        v_prev = v_now


class _D3Q19Solver:
    """D3Q19 MRT LBM solver for single-phase flow.

    Parameters
    ----------
    ti : module
        The taichi module.
    solid : np.ndarray
        Binary solid mask (1=solid, 0=pore), shape (nx, ny, nz).
    nu : float
        Kinematic viscosity in lattice units.
    sparse : bool
        Use Taichi sparse storage.
    """

    def __init__(self, ti, solid, nu=1 / 6, sparse=False):
        self._ti = ti
        self._nu = nu
        self._sparse = sparse
        nx, ny, nz = solid.shape
        self._nx, self._ny, self._nz = nx, ny, nz

        # Relaxation parameters
        self._tau_f = 3.0 * nu + 0.5
        s_v = 1.0 / self._tau_f
        s_other = 8.0 * (2.0 - s_v) / (8.0 - s_v)
        self._S_diag = np.array(
            [
                0,
                s_v,
                s_v,
                0,
                s_other,
                0,
                s_other,
                0,
                s_other,
                s_v,
                s_v,
                s_v,
                s_v,
                s_v,
                s_v,
                s_v,
                s_other,
                s_other,
                s_other,
            ],
            dtype=np.float32,
        )

        # Solid field
        self._solid = ti.field(ti.i8, shape=(nx, ny, nz))
        self._solid.from_numpy(solid.astype(np.int8))

        # Distribution and macroscopic fields
        if not sparse:
            self._f = ti.Vector.field(
                19,
                ti.f32,
                shape=(nx, ny, nz),
                layout=ti.Layout.SOA,
            )
            self._F = ti.Vector.field(
                19,
                ti.f32,
                shape=(nx, ny, nz),
                layout=ti.Layout.SOA,
            )
            self._rho = ti.field(ti.f32, shape=(nx, ny, nz))
            self._v = ti.Vector.field(
                3,
                ti.f32,
                shape=(nx, ny, nz),
            )
        else:
            self._f = ti.Vector.field(19, ti.f32)
            self._F = ti.Vector.field(19, ti.f32)
            self._rho = ti.field(ti.f32)
            self._v = ti.Vector.field(3, ti.f32)
            part = 3
            cell = ti.root.pointer(
                ti.ijk,
                (nx // part + 1, ny // part + 1, nz // part + 1),
            )
            cell.dense(
                ti.ijk,
                (part, part, part),
            ).place(self._rho, self._v, self._f, self._F)

        # Lattice vectors, weights, MRT matrices
        self._e = ti.Vector.field(3, ti.i32, shape=(19,))
        self._e_f = ti.Vector.field(3, ti.f32, shape=(19,))
        self._w = ti.field(ti.f32, shape=(19,))
        self._M = ti.field(ti.f32, shape=(19, 19))
        self._M_inv = ti.field(ti.f32, shape=(19, 19))
        self._S_dig = ti.Vector.field(19, ti.f32, shape=())

        # Load MRT matrices from precomputed NumPy arrays
        self._M.from_numpy(D3Q19_M)
        self._M_inv.from_numpy(D3Q19_M_inv)
        self._S_dig[None] = ti.Vector(self._S_diag.tolist())

        # Boundary condition state
        # mode: 0=periodic, 1=pressure (fixed rho)
        self._bc_mode = {
            "x0": 0,
            "x1": 0,
            "y0": 0,
            "y1": 0,
            "z0": 0,
            "z1": 0,
        }
        self._bc_rho = {
            "x0": 1.0,
            "x1": 1.0,
            "y0": 1.0,
            "y1": 1.0,
            "z0": 1.0,
            "z1": 1.0,
        }

    def set_bc_rho(self, face, rho):
        """Set fixed-pressure BC on a domain face.

        Parameters
        ----------
        face : str
            One of 'x0', 'x1', 'y0', 'y1', 'z0', 'z1'.
        rho : float
            Density (pressure) value.
        """
        self._bc_mode[face] = 1
        self._bc_rho[face] = float(rho)

    def init_fields(self):
        """Initialize lattice vectors, weights, and distribution fields."""
        self._init_lattice()
        self._init_distributions()
        self._compile_step_kernels()

    def step(self):
        """Advance by one LBM time step."""
        self._collide()
        self._stream()
        self._apply_bc()
        self._finalize()

    def get_velocity(self):
        """Extract velocity field as NumPy array, shape (nx,ny,nz,3)."""
        return self._v.to_numpy()

    def get_density(self):
        """Extract density field as NumPy array, shape (nx,ny,nz)."""
        return self._rho.to_numpy()

    # ── Taichi kernels ─────────────────────────────────────────────────

    def _init_lattice(self):
        """Set D3Q19 lattice velocities and weights."""
        ti = self._ti

        @ti.kernel
        def kernel(
            e: ti.template(),
            e_f: ti.template(),
            w: ti.template(),
        ):
            e[0] = ti.Vector([0, 0, 0])
            e[1] = ti.Vector([1, 0, 0])
            e[2] = ti.Vector([-1, 0, 0])
            e[3] = ti.Vector([0, 1, 0])
            e[4] = ti.Vector([0, -1, 0])
            e[5] = ti.Vector([0, 0, 1])
            e[6] = ti.Vector([0, 0, -1])
            e[7] = ti.Vector([1, 1, 0])
            e[8] = ti.Vector([-1, -1, 0])
            e[9] = ti.Vector([1, -1, 0])
            e[10] = ti.Vector([-1, 1, 0])
            e[11] = ti.Vector([1, 0, 1])
            e[12] = ti.Vector([-1, 0, -1])
            e[13] = ti.Vector([1, 0, -1])
            e[14] = ti.Vector([-1, 0, 1])
            e[15] = ti.Vector([0, 1, 1])
            e[16] = ti.Vector([0, -1, -1])
            e[17] = ti.Vector([0, 1, -1])
            e[18] = ti.Vector([0, -1, 1])
            for s in ti.static(range(19)):
                e_f[s] = ti.cast(e[s], ti.f32)
            w[0] = 1.0 / 3.0
            for s in ti.static(range(1, 7)):
                w[s] = 1.0 / 18.0
            for s in ti.static(range(7, 19)):
                w[s] = 1.0 / 36.0

        kernel(self._e, self._e_f, self._w)

    def _init_distributions(self):
        """Initialize density=1, velocity=0, equilibrium distributions."""
        ti = self._ti

        @ti.kernel
        def kernel(
            solid: ti.template(),
            rho: ti.template(),
            v: ti.template(),
            f: ti.template(),
            F: ti.template(),
            w: ti.template(),
            sparse: ti.template(),
        ):
            for i, j, k in solid:
                if (not sparse) or (solid[i, j, k] == 0):
                    rho[i, j, k] = 1.0
                    v[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
                    for s in ti.static(range(19)):
                        f[i, j, k][s] = w[s]
                        F[i, j, k][s] = w[s]

        kernel(
            self._solid,
            self._rho,
            self._v,
            self._f,
            self._F,
            self._w,
            self._sparse,
        )

    def _compile_step_kernels(self):
        """Pre-compile all per-step Taichi kernels once."""
        ti = self._ti
        LR = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17]

        @ti.kernel
        def _kern_collide(
            solid: ti.template(),
            rho: ti.template(),
            v: ti.template(),
            f: ti.template(),
            F: ti.template(),
            M: ti.template(),
            inv_M: ti.template(),
            S_dig: ti.template(),
        ):
            for i, j, k in rho:
                if solid[i, j, k] == 0:
                    m = ti.Vector([0.0] * 19)
                    for row in ti.static(range(19)):
                        for col in ti.static(range(19)):
                            m[row] += M[row, col] * F[i, j, k][col]
                    u = v[i, j, k]
                    rho_l = rho[i, j, k]
                    meq = ti.Vector([0.0] * 19)
                    meq[0] = rho_l
                    meq[3] = u[0]
                    meq[5] = u[1]
                    meq[7] = u[2]
                    meq[1] = u.dot(u)
                    meq[9] = 2 * u.x * u.x - u.y * u.y - u.z * u.z
                    meq[11] = u.y * u.y - u.z * u.z
                    meq[13] = u.x * u.y
                    meq[14] = u.y * u.z
                    meq[15] = u.x * u.z
                    m -= S_dig[None] * (m - meq)
                    f[i, j, k] = ti.Vector([0.0] * 19)
                    for row in ti.static(range(19)):
                        for col in ti.static(range(19)):
                            f[i, j, k][row] += inv_M[row, col] * m[col]

        @ti.kernel
        def _kern_stream(
            solid: ti.template(),
            f: ti.template(),
            F: ti.template(),
            e: ti.template(),
            nx_: int,
            ny_: int,
            nz_: int,
        ):
            for i in ti.grouped(f):
                if solid[i] == 0:
                    for s in ti.static(range(19)):
                        ip = i + e[s]
                        if ip[0] < 0:
                            ip[0] = nx_ - 1
                        if ip[0] >= nx_:
                            ip[0] = 0
                        if ip[1] < 0:
                            ip[1] = ny_ - 1
                        if ip[1] >= ny_:
                            ip[1] = 0
                        if ip[2] < 0:
                            ip[2] = nz_ - 1
                        if ip[2] >= nz_:
                            ip[2] = 0
                        if solid[ip] == 0:
                            F[ip][s] = f[i][s]
                        else:
                            F[i][LR[s]] = f[i][s]

        @ti.kernel
        def _kern_finalize(
            solid: ti.template(),
            f: ti.template(),
            F: ti.template(),
            rho: ti.template(),
            v: ti.template(),
            e_f: ti.template(),
        ):
            for i in ti.grouped(rho):
                if solid[i] == 0:
                    f[i] = F[i]
                    rho[i] = f[i].sum()
                    v[i] = ti.Vector([0.0, 0.0, 0.0])
                    for s in ti.static(range(19)):
                        v[i] += e_f[s] * f[i][s]
                    v[i] /= rho[i]
                else:
                    rho[i] = 1.0
                    v[i] = ti.Vector([0.0, 0.0, 0.0])

        @ti.kernel
        def _kern_bc_x(
            solid_: ti.template(),
            F_: ti.template(),
            v_: ti.template(),
            w_: ti.template(),
            e_f_: ti.template(),
            idx_: int,
            nb_: int,
            rho_bc_: float,
            ny_: int,
            nz_: int,
        ):
            for j, k in ti.ndrange((0, ny_), (0, nz_)):
                if solid_[idx_, j, k] == 0:
                    u = v_[idx_, j, k]
                    if solid_[nb_, j, k] > 0:
                        u = v_[nb_, j, k]
                    for s in ti.static(range(19)):
                        eu = e_f_[s].dot(u)
                        uv = u.dot(u)
                        F_[idx_, j, k][s] = (
                            w_[s] * rho_bc_ * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * uv)
                        )

        @ti.kernel
        def _kern_bc_y(
            solid_: ti.template(),
            F_: ti.template(),
            v_: ti.template(),
            w_: ti.template(),
            e_f_: ti.template(),
            idx_: int,
            nb_: int,
            rho_bc_: float,
            nx_: int,
            nz_: int,
        ):
            for i, k in ti.ndrange((0, nx_), (0, nz_)):
                if solid_[i, idx_, k] == 0:
                    u = v_[i, idx_, k]
                    if solid_[i, nb_, k] > 0:
                        u = v_[i, nb_, k]
                    for s in ti.static(range(19)):
                        eu = e_f_[s].dot(u)
                        uv = u.dot(u)
                        F_[i, idx_, k][s] = (
                            w_[s] * rho_bc_ * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * uv)
                        )

        @ti.kernel
        def _kern_bc_z(
            solid_: ti.template(),
            F_: ti.template(),
            v_: ti.template(),
            w_: ti.template(),
            e_f_: ti.template(),
            idx_: int,
            nb_: int,
            rho_bc_: float,
            nx_: int,
            ny_: int,
        ):
            for i, j in ti.ndrange((0, nx_), (0, ny_)):
                if solid_[i, j, idx_] == 0:
                    u = v_[i, j, idx_]
                    if solid_[i, j, nb_] > 0:
                        u = v_[i, j, nb_]
                    for s in ti.static(range(19)):
                        eu = e_f_[s].dot(u)
                        uv = u.dot(u)
                        F_[i, j, idx_][s] = (
                            w_[s] * rho_bc_ * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * uv)
                        )

        self._kern_collide = _kern_collide
        self._kern_stream = _kern_stream
        self._kern_finalize = _kern_finalize
        self._kern_bc_x = _kern_bc_x
        self._kern_bc_y = _kern_bc_y
        self._kern_bc_z = _kern_bc_z

    def _collide(self):
        """MRT collision operator."""
        self._kern_collide(
            self._solid,
            self._rho,
            self._v,
            self._f,
            self._F,
            self._M,
            self._M_inv,
            self._S_dig,
        )

    def _stream(self):
        """Stream distributions with bounce-back on solids."""
        self._kern_stream(
            self._solid,
            self._f,
            self._F,
            self._e,
            self._nx,
            self._ny,
            self._nz,
        )

    def _apply_bc(self):
        """Apply pressure BCs (fixed rho) on active faces."""
        nx, ny, nz = self._nx, self._ny, self._nz
        args = (self._solid, self._F, self._v, self._w, self._e_f)
        face_calls = {
            "x0": lambda: self._kern_bc_x(*args, 0, 1, self._bc_rho["x0"], ny, nz),
            "x1": lambda: self._kern_bc_x(
                *args, nx - 1, nx - 2, self._bc_rho["x1"], ny, nz
            ),
            "y0": lambda: self._kern_bc_y(*args, 0, 1, self._bc_rho["y0"], nx, nz),
            "y1": lambda: self._kern_bc_y(
                *args, ny - 1, ny - 2, self._bc_rho["y1"], nx, nz
            ),
            "z0": lambda: self._kern_bc_z(*args, 0, 1, self._bc_rho["z0"], nx, ny),
            "z1": lambda: self._kern_bc_z(
                *args, nz - 1, nz - 2, self._bc_rho["z1"], nx, ny
            ),
        }
        for face, mode in self._bc_mode.items():
            if mode == 1:
                face_calls[face]()

    def _finalize(self):
        """Copy F -> f, recompute macroscopic density and velocity."""
        self._kern_finalize(
            self._solid,
            self._f,
            self._F,
            self._rho,
            self._v,
            self._e_f,
        )
