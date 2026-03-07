# D3Q7 BGK Lattice Boltzmann solver for passive scalar diffusion.
import time

import numpy as np
from loguru import logger

from ._taichi_helpers import ensure_taichi


# Dirichlet BCs: Δc = 1 simplifies Fick's law (D_eff = J * L / Δc)
_C_IN = 1.0
_C_OUT = 0.0

# Map axis int to (inlet_face, outlet_face) pairs
_AXIS_FACES = {
    0: ("x0", "x1"),
    1: ("y0", "y1"),
    2: ("z0", "z1"),
}


def _build_solver(solid, axis, D, sparse):
    """Construct and initialize a D3Q7 BGK diffusion solver.

    Parameters
    ----------
    solid : np.ndarray
        Binary solid mask (1=solid, 0=pore).
    axis : int
        Flow axis (0=x, 1=y, 2=z).
    D : float
        Bulk diffusivity in lattice units.
    sparse : bool
        Use Taichi sparse storage.

    Returns
    -------
    solver : _D3Q7Solver
        Initialized solver ready to step.
    """
    ti = ensure_taichi()
    solver = _D3Q7Solver(ti, solid, D=D, sparse=sparse)
    inlet, outlet = _AXIS_FACES[axis]
    solver.set_bc(inlet, _C_IN)
    solver.set_bc(outlet, _C_OUT)
    solver.init_fields()
    return solver


def solve_diffusion(
    solid, axis, D=0.25, n_steps=100_000, tol=1e-2, log_every=500, sparse=False
):
    """Run LBM diffusion to steady state and return arrays.

    Parameters
    ----------
    solid : np.ndarray, shape (nx, ny, nz), dtype int8
        Binary solid mask (1=solid, 0=pore).
    axis : int
        Flow axis (0=x, 1=y, 2=z).
    D : float
        Bulk diffusivity in lattice units. Default 0.25.
        Relaxation time: tau_D = 4*D + 0.5 (D3Q7 cs^2 = 1/4).
    n_steps : int
        Maximum LBM iterations.
    tol : float or None
        Convergence tolerance on delta|c|/|c|. None disables
        early stopping.
    log_every : int
        Log convergence progress every this many steps.
    sparse : bool
        Use Taichi pointer-backed sparse storage.

    Returns
    -------
    c : np.ndarray
        Steady-state concentration field.
    J_mean : float
        Mean diffusive flux through the domain midplane.
    """
    solver = _build_solver(solid, axis, D, sparse)
    _run_to_convergence(solver, n_steps, tol, log_every)
    c = solver.get_concentration()
    J_mean = solver.compute_flux(axis)
    return c, J_mean


def _run_to_convergence(solver, n_steps, tol, log_every):
    """Step the solver until convergence or max steps."""
    t_start = time.time()
    c_prev = None
    for step in range(n_steps + 1):
        solver.step()
        if step % log_every != 0:
            continue
        elapsed = time.time() - t_start
        c_now = solver.get_concentration()
        if c_prev is not None:
            c_total = np.sum(np.abs(c_now))
            c_change = np.sum(np.abs(c_now - c_prev))
            ratio = c_change / c_total if c_total > 0 else 0.0
            logger.info(
                f"Step {step:>6d}/{n_steps}  "
                f"|c|={c_total:.4e}  "
                f"delta={ratio:.2e}  "
                f"elapsed={elapsed:.1f}s"
            )
            if tol is not None and c_total > 0 and ratio < tol:
                logger.info(
                    f"Converged at step {step} (delta|c|/|c|={ratio:.2e} < tol={tol:.2e})"
                )
                return
        c_prev = c_now


class _D3Q7Solver:
    """D3Q7 BGK LBM solver for diffusion, backed by Taichi kernels.

    Parameters
    ----------
    ti : module
        The taichi module (passed in to avoid top-level import).
    solid : np.ndarray
        Binary solid mask (1=solid, 0=pore), shape (nx, ny, nz).
    D : float
        Bulk diffusivity in lattice units.
    sparse : bool
        Use Taichi sparse storage.
    """

    def __init__(self, ti, solid, D=0.25, sparse=False):
        self._ti = ti
        self._D = D
        self._sparse = sparse
        self._tau_D = 4.0 * D + 0.5
        nx, ny, nz = solid.shape
        self._nx, self._ny, self._nz = nx, ny, nz

        # Solid field
        self._solid = ti.field(ti.i8, shape=(nx, ny, nz))
        self._solid.from_numpy(solid.astype(np.int8))

        # Distribution fields
        if not sparse:
            self._g = ti.Vector.field(
                7,
                ti.f32,
                shape=(nx, ny, nz),
                layout=ti.Layout.SOA,
            )
            self._G = ti.Vector.field(
                7,
                ti.f32,
                shape=(nx, ny, nz),
                layout=ti.Layout.SOA,
            )
            self._c = ti.field(ti.f32, shape=(nx, ny, nz))
        else:
            self._g = ti.Vector.field(7, ti.f32)
            self._G = ti.Vector.field(7, ti.f32)
            self._c = ti.field(ti.f32)
            part = 3
            cell = ti.root.pointer(
                ti.ijk,
                (nx // part + 1, ny // part + 1, nz // part + 1),
            )
            cell.dense(
                ti.ijk,
                (part, part, part),
            ).place(self._c, self._g, self._G)

        # Lattice vectors and weights (set in _init_lattice kernel)
        self._e = ti.Vector.field(3, ti.i32, shape=(7,))
        self._w = ti.field(ti.f32, shape=(7,))

        # Boundary condition state: 0=periodic, 1=fixed concentration
        self._bc_mode = {
            "x0": 0,
            "x1": 0,
            "y0": 0,
            "y1": 0,
            "z0": 0,
            "z1": 0,
        }
        self._bc_val = {
            "x0": 0.0,
            "x1": 0.0,
            "y0": 0.0,
            "y1": 0.0,
            "z0": 0.0,
            "z1": 0.0,
        }

    def set_bc(self, face, value):
        """Set fixed-concentration BC on a domain face.

        Parameters
        ----------
        face : str
            One of 'x0', 'x1', 'y0', 'y1', 'z0', 'z1'.
        value : float
            Concentration value.
        """
        self._bc_mode[face] = 1
        self._bc_val[face] = float(value)

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

    def get_concentration(self):
        """Extract concentration field as NumPy array."""
        return self._c.to_numpy()

    def compute_flux(self, axis):
        """Compute diffusive flux at the domain midplane.

        Uses Fick's law (J = -D * dc/dx) on the concentration field
        rather than distribution moments, which are unreliable at
        Dirichlet boundary faces.

        Parameters
        ----------
        axis : int
            0=x, 1=y, 2=z.

        Returns
        -------
        J_mean : float
            Mean diffusive flux through the midplane cross-section.
        """
        c_np = self._c.to_numpy()
        mid = c_np.shape[axis] // 2
        slc_hi = [slice(None)] * 3
        slc_lo = [slice(None)] * 3
        slc_hi[axis] = mid
        slc_lo[axis] = mid - 1
        dc = c_np[tuple(slc_hi)] - c_np[tuple(slc_lo)]
        solid_np = self._solid.to_numpy()
        # Mask out solid voxels on both sides of the cross-section
        pore_mask = (solid_np[tuple(slc_hi)] == 0) & (solid_np[tuple(slc_lo)] == 0)
        J = -self._D * dc
        J[~pore_mask] = 0.0
        return float(np.mean(J))

    # ── Taichi kernels ─────────────────────────────────────────────────

    def _init_lattice(self):
        """Set D3Q7 lattice velocities and weights."""
        ti = self._ti

        @ti.kernel
        def kernel(
            e: ti.template(),
            w: ti.template(),
        ):
            e[0] = ti.Vector([0, 0, 0])
            e[1] = ti.Vector([1, 0, 0])
            e[2] = ti.Vector([-1, 0, 0])
            e[3] = ti.Vector([0, 1, 0])
            e[4] = ti.Vector([0, -1, 0])
            e[5] = ti.Vector([0, 0, 1])
            e[6] = ti.Vector([0, 0, -1])
            w[0] = 1.0 / 4.0
            for s in ti.static(range(1, 7)):
                w[s] = 1.0 / 8.0

        kernel(self._e, self._w)

    def _init_distributions(self):
        """Initialize concentration to 0.5 everywhere in pore space."""
        ti = self._ti

        @ti.kernel
        def kernel(
            solid: ti.template(),
            c: ti.template(),
            g: ti.template(),
            G: ti.template(),
            w: ti.template(),
            sparse: ti.template(),
        ):
            for i, j, k in solid:
                if (not sparse) or (solid[i, j, k] == 0):
                    c[i, j, k] = 0.5
                    for s in ti.static(range(7)):
                        g[i, j, k][s] = w[s] * 0.5
                        G[i, j, k][s] = w[s] * 0.5

        kernel(
            self._solid,
            self._c,
            self._g,
            self._G,
            self._w,
            self._sparse,
        )

    def _compile_step_kernels(self):
        """Pre-compile all per-step Taichi kernels once."""
        ti = self._ti
        LR = [0, 2, 1, 4, 3, 6, 5]

        @ti.kernel
        def _kern_collide(
            solid: ti.template(),
            c: ti.template(),
            g: ti.template(),
            w: ti.template(),
            tau_D: float,
        ):
            for i, j, k in c:
                if solid[i, j, k] == 0:
                    c_local = c[i, j, k]
                    for s in ti.static(range(7)):
                        g[i, j, k][s] -= (g[i, j, k][s] - w[s] * c_local) / tau_D

        @ti.kernel
        def _kern_stream(
            solid: ti.template(),
            g: ti.template(),
            G: ti.template(),
            e: ti.template(),
            nx_: int,
            ny_: int,
            nz_: int,
        ):
            for i in ti.grouped(g):
                if solid[i] == 0:
                    for s in ti.static(range(7)):
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
                            G[ip][s] = g[i][s]
                        else:
                            G[i][LR[s]] = g[i][s]

        @ti.kernel
        def _kern_finalize(
            solid: ti.template(),
            g: ti.template(),
            G: ti.template(),
            c: ti.template(),
        ):
            for i in ti.grouped(c):
                if solid[i] == 0:
                    g[i] = G[i]
                    c[i] = g[i].sum()
                else:
                    c[i] = 0.0

        @ti.kernel
        def _kern_bc_x(
            solid_: ti.template(),
            G_: ti.template(),
            w_: ti.template(),
            idx_: int,
            c_bc_: float,
            ny_: int,
            nz_: int,
        ):
            for j, k in ti.ndrange((0, ny_), (0, nz_)):
                if solid_[idx_, j, k] == 0:
                    for s in ti.static(range(7)):
                        G_[idx_, j, k][s] = w_[s] * c_bc_

        @ti.kernel
        def _kern_bc_y(
            solid_: ti.template(),
            G_: ti.template(),
            w_: ti.template(),
            idx_: int,
            c_bc_: float,
            nx_: int,
            nz_: int,
        ):
            for i, k in ti.ndrange((0, nx_), (0, nz_)):
                if solid_[i, idx_, k] == 0:
                    for s in ti.static(range(7)):
                        G_[i, idx_, k][s] = w_[s] * c_bc_

        @ti.kernel
        def _kern_bc_z(
            solid_: ti.template(),
            G_: ti.template(),
            w_: ti.template(),
            idx_: int,
            c_bc_: float,
            nx_: int,
            ny_: int,
        ):
            for i, j in ti.ndrange((0, nx_), (0, ny_)):
                if solid_[i, j, idx_] == 0:
                    for s in ti.static(range(7)):
                        G_[i, j, idx_][s] = w_[s] * c_bc_

        self._kern_collide = _kern_collide
        self._kern_stream = _kern_stream
        self._kern_finalize = _kern_finalize
        self._kern_bc_x = _kern_bc_x
        self._kern_bc_y = _kern_bc_y
        self._kern_bc_z = _kern_bc_z

    def _collide(self):
        """BGK collision: relax toward equilibrium."""
        self._kern_collide(self._solid, self._c, self._g, self._w, self._tau_D)

    def _stream(self):
        """Stream distributions with bounce-back on solids."""
        self._kern_stream(
            self._solid,
            self._g,
            self._G,
            self._e,
            self._nx,
            self._ny,
            self._nz,
        )

    def _apply_bc(self):
        """Apply fixed-concentration Dirichlet BCs on active faces."""
        nx, ny, nz = self._nx, self._ny, self._nz
        args = (self._solid, self._G, self._w)
        face_calls = {
            "x0": lambda: self._kern_bc_x(*args, 0, self._bc_val["x0"], ny, nz),
            "x1": lambda: self._kern_bc_x(*args, nx - 1, self._bc_val["x1"], ny, nz),
            "y0": lambda: self._kern_bc_y(*args, 0, self._bc_val["y0"], nx, nz),
            "y1": lambda: self._kern_bc_y(*args, ny - 1, self._bc_val["y1"], nx, nz),
            "z0": lambda: self._kern_bc_z(*args, 0, self._bc_val["z0"], nx, ny),
            "z1": lambda: self._kern_bc_z(*args, nz - 1, self._bc_val["z1"], nx, ny),
        }
        for face, mode in self._bc_mode.items():
            if mode == 1:
                face_calls[face]()

    def _finalize(self):
        """Copy G -> g and recompute macroscopic concentration."""
        self._kern_finalize(self._solid, self._g, self._G, self._c)
