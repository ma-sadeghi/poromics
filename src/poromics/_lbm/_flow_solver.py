# D3Q19 MRT Lattice Boltzmann solver for single-phase incompressible flow.
# Based on taichi_LBM3D by Yi-Jie Huang (https://github.com/yjhp1016/taichi_LBM3D).
import numpy as np

from ._lattice import D3Q19_LR, D3Q19_M, D3Q19_M_inv


def _build_relaxation_diag(nu):
    """Build the diagonal MRT relaxation vector.

    Parameters
    ----------
    nu : float
        Kinematic viscosity in lattice units.

    Returns
    -------
    S_diag : np.ndarray, shape (19,), dtype float32
        Diagonal entries of the relaxation matrix S.

    Notes
    -----
    Moment ordering follows the standard D3Q19 MRT convention:
    [rho, e, eps, jx, qx, jy, qy, jz, qz,
     3pxx, 3pixx, pww, piww, pxy, pyz, pxz, mx, my, mz]
    """
    tau_f = 3.0 * nu + 0.5
    s_v = 1.0 / tau_f
    s_other = 8.0 * (2.0 - s_v) / (8.0 - s_v)
    # fmt: off
    return np.array([
        0,        # rho   (conserved)
        s_v,      # e     (energy)
        s_v,      # eps   (energy square)
        0,        # jx    (conserved)
        s_other,  # qx    (energy flux)
        0,        # jy    (conserved)
        s_other,  # qy    (energy flux)
        0,        # jz    (conserved)
        s_other,  # qz    (energy flux)
        s_v,      # 3pxx  (stress)
        s_v,      # 3pixx
        s_v,      # pww   (stress)
        s_v,      # piww
        s_v,      # pxy   (stress)
        s_v,      # pyz   (stress)
        s_v,      # pxz   (stress)
        s_other,  # mx    (kinetic)
        s_other,  # my    (kinetic)
        s_other,  # mz    (kinetic)
    ], dtype=np.float32)
    # fmt: on


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
        self._sparse = sparse
        nx, ny, nz = solid.shape
        self._nx, self._ny, self._nz = nx, ny, nz

        self._alloc_fields(ti, solid, sparse)
        self._load_constants(ti, nu)
        self._init_bc_state()

    def _alloc_fields(self, ti, solid, sparse):
        """Allocate Taichi fields for distributions and macroscopic vars."""
        nx, ny, nz = self._nx, self._ny, self._nz

        self._solid = ti.field(ti.i8, shape=(nx, ny, nz))
        self._solid.from_numpy(solid.astype(np.int8))

        if not sparse:
            self._f = ti.Vector.field(19, ti.f32, shape=(nx, ny, nz), layout=ti.Layout.SOA)
            self._F = ti.Vector.field(19, ti.f32, shape=(nx, ny, nz), layout=ti.Layout.SOA)
            self._rho = ti.field(ti.f32, shape=(nx, ny, nz))
            self._v = ti.Vector.field(3, ti.f32, shape=(nx, ny, nz))
        else:
            self._f = ti.Vector.field(19, ti.f32)
            self._F = ti.Vector.field(19, ti.f32)
            self._rho = ti.field(ti.f32)
            self._v = ti.Vector.field(3, ti.f32)
            part = 3
            cell = ti.root.pointer(ti.ijk, (nx // part + 1, ny // part + 1, nz // part + 1))
            cell.dense(ti.ijk, (part, part, part)).place(self._rho, self._v, self._f, self._F)

    def _load_constants(self, ti, nu):
        """Load lattice vectors, weights, and MRT matrices into fields."""
        self._e = ti.Vector.field(3, ti.i32, shape=(19,))
        self._e_f = ti.Vector.field(3, ti.f32, shape=(19,))
        self._w = ti.field(ti.f32, shape=(19,))
        self._M = ti.field(ti.f32, shape=(19, 19))
        self._M_inv = ti.field(ti.f32, shape=(19, 19))
        self._S_dig = ti.Vector.field(19, ti.f32, shape=())

        self._M.from_numpy(D3Q19_M)
        self._M_inv.from_numpy(D3Q19_M_inv)
        S_diag = _build_relaxation_diag(nu)
        self._S_dig[None] = ti.Vector(S_diag.tolist())

    def _init_bc_state(self):
        """Initialize boundary condition mode and values for all 6 faces."""
        faces = ("x0", "x1", "y0", "y1", "z0", "z1")
        # mode: 0=periodic, 1=pressure (fixed rho)
        self._bc_mode = {f: 0 for f in faces}
        self._bc_rho = {f: 1.0 for f in faces}

    # ── Public interface ──────────────────────────────────────────────

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
        """Advance by one LBM time step.

        Step order: finalize+collide (F -> rho, v, f) -> stream (f -> F)
        -> bc (F). The fused finalize+collide kernel reads F once and
        produces both macroscopic fields and post-collision distributions,
        saving one full-field memory pass.
        """
        self._finalize_collide(
            self._solid, self._rho, self._v, self._f, self._F,
            self._M, self._M_inv, self._S_dig, self._e_f,
        )  # fmt: skip
        self._stream(
            self._solid, self._f, self._F, self._e,
            self._nx, self._ny, self._nz,
        )  # fmt: skip
        self._apply_bc()

    def _apply_bc(self):
        """Apply equilibrium pressure BCs on active faces."""
        nx, ny, nz = self._nx, self._ny, self._nz
        args = (self._solid, self._F, self._v, self._w, self._e_f)
        for face, mode in self._bc_mode.items():
            if mode != 1:
                continue
            rho = self._bc_rho[face]
            if face == "x0":
                self._bc_x(*args, 0, 1, rho, ny, nz)
            elif face == "x1":
                self._bc_x(*args, nx - 1, nx - 2, rho, ny, nz)
            elif face == "y0":
                self._bc_y(*args, 0, 1, rho, nx, nz)
            elif face == "y1":
                self._bc_y(*args, ny - 1, ny - 2, rho, nx, nz)
            elif face == "z0":
                self._bc_z(*args, 0, 1, rho, nx, ny)
            elif face == "z1":
                self._bc_z(*args, nz - 1, nz - 2, rho, nx, ny)

    def get_velocity(self):
        """Extract velocity field as NumPy array, shape (nx,ny,nz,3)."""
        return self._v.to_numpy()

    def get_density(self):
        """Extract density field as NumPy array, shape (nx,ny,nz)."""
        return self._rho.to_numpy()

    # ── Taichi kernels ────────────────────────────────────────────────

    def _init_lattice(self):
        """Set D3Q19 lattice velocities and weights."""
        ti = self._ti

        @ti.kernel
        def kernel(e: ti.template(), e_f: ti.template(), w: ti.template()):
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
            solid: ti.template(), rho: ti.template(), v: ti.template(),
            f: ti.template(), F: ti.template(), w: ti.template(),
            sparse: ti.template(),
        ):  # fmt: skip
            for i, j, k in solid:
                if (not sparse) or (solid[i, j, k] == 0):
                    rho[i, j, k] = 1.0
                    v[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
                    for s in ti.static(range(19)):
                        f[i, j, k][s] = w[s]
                        F[i, j, k][s] = w[s]

        kernel(self._solid, self._rho, self._v, self._f, self._F, self._w, self._sparse)

    def _compile_step_kernels(self):
        """Pre-compile all per-step Taichi kernels once."""
        ti = self._ti
        LR = D3Q19_LR

        # ── Fused finalize + collision ────────────────────────────────

        @ti.kernel
        def finalize_collide(
            solid: ti.template(), rho: ti.template(), v: ti.template(),
            f: ti.template(), F: ti.template(), M: ti.template(),
            inv_M: ti.template(), S_dig: ti.template(), e_f: ti.template(),
        ):  # fmt: skip
            for i in ti.grouped(rho):
                if solid[i] == 0:
                    # Compute macroscopic rho and u from F
                    rho_l = F[i].sum()
                    rho[i] = rho_l
                    u = ti.Vector([0.0, 0.0, 0.0])
                    for s in ti.static(range(19)):
                        u += e_f[s] * F[i][s]
                    u /= rho_l
                    v[i] = u
                    # MRT collision: transform to moment space, relax,
                    # transform back to velocity space
                    m = ti.Vector([0.0] * 19)
                    for row in ti.static(range(19)):
                        for col in ti.static(range(19)):
                            m[row] += M[row, col] * F[i][col]
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
                    f[i] = ti.Vector([0.0] * 19)
                    for row in ti.static(range(19)):
                        for col in ti.static(range(19)):
                            f[i][row] += inv_M[row, col] * m[col]
                else:
                    rho[i] = 1.0
                    v[i] = ti.Vector([0.0, 0.0, 0.0])

        # ── Streaming with periodic BCs and bounce-back ───────────────

        @ti.kernel
        def stream(
            solid: ti.template(), f: ti.template(), F: ti.template(),
            e: ti.template(), nx_: int, ny_: int, nz_: int,
        ):  # fmt: skip
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

        # ── Equilibrium pressure BCs (one kernel per axis) ────────────

        @ti.kernel
        def bc_x(
            solid_: ti.template(), F_: ti.template(), v_: ti.template(),
            w_: ti.template(), e_f_: ti.template(), idx_: int, nb_: int,
            rho_bc_: float, ny_: int, nz_: int,
        ):  # fmt: skip
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
        def bc_y(
            solid_: ti.template(), F_: ti.template(), v_: ti.template(),
            w_: ti.template(), e_f_: ti.template(), idx_: int, nb_: int,
            rho_bc_: float, nx_: int, nz_: int,
        ):  # fmt: skip
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
        def bc_z(
            solid_: ti.template(), F_: ti.template(), v_: ti.template(),
            w_: ti.template(), e_f_: ti.template(), idx_: int, nb_: int,
            rho_bc_: float, nx_: int, ny_: int
        ):  # fmt: skip
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

        self._finalize_collide = finalize_collide
        self._stream = stream
        self._bc_x = bc_x
        self._bc_y = bc_y
        self._bc_z = bc_z
