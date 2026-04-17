# D3Q7 BGK Lattice Boltzmann solver for passive scalar diffusion.
# Based on taichi_LBM3D by Yi-Jie Huang (https://github.com/yjhp1016/taichi_LBM3D).
import numpy as np

from ._lattice import D3Q7_LR


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

        self._alloc_fields(ti, solid, sparse)
        self._load_constants(ti)
        self._init_bc_state()

    def _alloc_fields(self, ti, solid, sparse):
        """Allocate Taichi fields for distributions and concentration."""
        nx, ny, nz = self._nx, self._ny, self._nz

        if not sparse:
            self._solid = ti.field(ti.i8, shape=(nx, ny, nz))
            self._solid.from_numpy(solid.astype(np.int8))
            self._g = ti.Vector.field(7, ti.f32, shape=(nx, ny, nz), layout=ti.Layout.SOA)
            self._G = ti.Vector.field(7, ti.f32, shape=(nx, ny, nz), layout=ti.Layout.SOA)
            self._c = ti.field(ti.f32, shape=(nx, ny, nz))
        else:
            # Pointer SNode allocates in blocks of `part`, so fields are
            # padded up to the next multiple. `_solid` must match that
            # padded shape with padding marked solid — otherwise kernels
            # iterating over `grouped(c)` hit activated padding cells and
            # treat them as pore because `solid[i]` reads beyond the
            # dense-backed range.
            part = 3
            px = (nx // part + 1) * part
            py = (ny // part + 1) * part
            pz = (nz // part + 1) * part
            solid_padded = np.ones((px, py, pz), dtype=np.int8)
            solid_padded[:nx, :ny, :nz] = solid.astype(np.int8)
            self._solid = ti.field(ti.i8, shape=(px, py, pz))
            self._solid.from_numpy(solid_padded)
            self._g = ti.Vector.field(7, ti.f32)
            self._G = ti.Vector.field(7, ti.f32)
            self._c = ti.field(ti.f32)
            cell = ti.root.pointer(ti.ijk, (nx // part + 1, ny // part + 1, nz // part + 1))
            cell.dense(ti.ijk, (part, part, part)).place(self._c, self._g, self._G)

    def _load_constants(self, ti):
        """Load lattice vectors and weights into Taichi fields."""
        self._e = ti.Vector.field(3, ti.i32, shape=(7,))
        self._w = ti.field(ti.f32, shape=(7,))

    def _init_bc_state(self):
        """Initialize boundary condition mode and values for all 6 faces."""
        faces = ("x0", "x1", "y0", "y1", "z0", "z1")
        # mode: 0=periodic, 1=fixed concentration
        self._bc_mode = {f: 0 for f in faces}
        self._bc_val = {f: 0.0 for f in faces}

    # ── Public interface ──────────────────────────────────────────────

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
        """Advance by one LBM time step.

        Step order: finalize+collide (G -> c, g) -> stream (g -> G)
        -> bc (G). The fused finalize+collide kernel reads G once and
        produces both the concentration field and post-collision
        distributions, saving one full-field memory pass.
        """
        self._finalize_collide(
            self._solid, self._c, self._g, self._G,
            self._w, self._tau_D,
        )  # fmt: skip
        self._stream(
            self._solid, self._g, self._G, self._e,
            self._nx, self._ny, self._nz,
        )  # fmt: skip
        self._apply_bc()

    def _apply_bc(self):
        """Apply fixed-concentration Dirichlet BCs on active faces."""
        nx, ny, nz = self._nx, self._ny, self._nz
        args = (self._solid, self._G, self._w)
        for face, mode in self._bc_mode.items():
            if mode != 1:
                continue
            val = self._bc_val[face]
            if face == "x0":
                self._bc_x(*args, 0, val, ny, nz)
            elif face == "x1":
                self._bc_x(*args, nx - 1, val, ny, nz)
            elif face == "y0":
                self._bc_y(*args, 0, val, nx, nz)
            elif face == "y1":
                self._bc_y(*args, ny - 1, val, nx, nz)
            elif face == "z0":
                self._bc_z(*args, 0, val, nx, ny)
            elif face == "z1":
                self._bc_z(*args, nz - 1, val, nx, ny)

    def get_concentration(self):
        """Extract concentration field as NumPy array."""
        # Sparse pointer SNode allocates in blocks of `part` voxels per axis,
        # so `_c` may be larger than (nx, ny, nz). Trim back to the image.
        nx, ny, nz = self._nx, self._ny, self._nz
        return self._c.to_numpy()[:nx, :ny, :nz]

    def compute_flux(self, axis, n_slices=5):
        """Compute diffusive flux averaged across interior slices.

        Uses Fick's law (J = -D * dc/dx) on the concentration field
        rather than distribution moments, which are unreliable at
        Dirichlet boundary faces. Averaging across several interior
        slices makes the estimate robust to local noise; we sample
        ``n_slices`` slices evenly spaced in the middle 60% of the
        domain, which excludes boundary-layer artifacts from the
        inlet/outlet faces.

        Parameters
        ----------
        axis : int
            0=x, 1=y, 2=z.
        n_slices : int
            Number of interior slices to average. Default 5. Use 1 to
            recover the single-midplane estimate.

        Returns
        -------
        J_mean : float
            Mean diffusive flux through the averaged cross-section.
        """
        if n_slices < 1:
            raise ValueError(f"n_slices must be >= 1, got {n_slices}")
        nx, ny, nz = self._nx, self._ny, self._nz
        c_np = self._c.to_numpy()[:nx, :ny, :nz]
        solid_np = self._solid.to_numpy()[:nx, :ny, :nz]
        L = c_np.shape[axis]
        # Need slc_lo >= 1 and slc_hi <= L-1 (each face is a Dirichlet BC).
        # Sample evenly across the central 60% of the domain.
        lo, hi = int(0.2 * L), int(0.8 * L)
        # Guard tiny domains: keep at least one valid slice.
        lo = max(lo, 1)
        hi = max(hi, lo + 1)
        if n_slices == 1:
            indices = [(lo + hi) // 2]
        else:
            indices = np.linspace(lo, hi - 1, n_slices, dtype=int).tolist()

        fluxes = []
        for mid in indices:
            slc_hi = [slice(None)] * 3
            slc_lo = [slice(None)] * 3
            slc_hi[axis] = mid
            slc_lo[axis] = mid - 1
            dc = c_np[tuple(slc_hi)] - c_np[tuple(slc_lo)]
            pore_mask = (
                (solid_np[tuple(slc_hi)] == 0) & (solid_np[tuple(slc_lo)] == 0)
            )
            J = -self._D * dc
            J[~pore_mask] = 0.0
            fluxes.append(float(np.mean(J)))
        return float(np.mean(fluxes))

    # ── Taichi kernels ────────────────────────────────────────────────

    def _init_lattice(self):
        """Set D3Q7 lattice velocities and weights."""
        ti = self._ti

        @ti.kernel
        def kernel(e: ti.template(), w: ti.template()):
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
            solid: ti.template(), c: ti.template(), g: ti.template(),
            G: ti.template(), w: ti.template(), sparse: ti.template(),
        ):  # fmt: skip
            for i, j, k in solid:
                if (not sparse) or (solid[i, j, k] == 0):
                    c[i, j, k] = 0.5
                    for s in ti.static(range(7)):
                        g[i, j, k][s] = w[s] * 0.5
                        G[i, j, k][s] = w[s] * 0.5

        kernel(self._solid, self._c, self._g, self._G, self._w, self._sparse)

    def _compile_step_kernels(self):
        """Pre-compile all per-step Taichi kernels once."""
        ti = self._ti
        LR = D3Q7_LR

        # ── Fused finalize + BGK collision ────────────────────────────

        @ti.kernel
        def finalize_collide(
            solid: ti.template(), c: ti.template(), g: ti.template(),
            G: ti.template(), w: ti.template(), tau_D: float,
        ):  # fmt: skip
            for i in ti.grouped(c):
                if solid[i] == 0:
                    # Finalize: compute c from post-stream distributions
                    c_local = G[i].sum()
                    c[i] = c_local
                    # BGK collision on the distributions
                    for s in ti.static(range(7)):
                        g[i][s] = G[i][s] - (G[i][s] - w[s] * c_local) / tau_D
                else:
                    c[i] = 0.0

        # ── Streaming with periodic BCs and bounce-back ───────────────

        @ti.kernel
        def stream(
            solid: ti.template(), g: ti.template(), G: ti.template(),
            e: ti.template(), nx_: int, ny_: int, nz_: int,
        ):  # fmt: skip
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

        # ── Dirichlet BCs (one kernel per axis) ──────────────────────

        @ti.kernel
        def bc_x(
            solid_: ti.template(), G_: ti.template(), w_: ti.template(),
            idx_: int, c_bc_: float, ny_: int, nz_: int,
        ):  # fmt: skip
            for j, k in ti.ndrange((0, ny_), (0, nz_)):
                if solid_[idx_, j, k] == 0:
                    for s in ti.static(range(7)):
                        G_[idx_, j, k][s] = w_[s] * c_bc_

        @ti.kernel
        def bc_y(
            solid_: ti.template(), G_: ti.template(), w_: ti.template(),
            idx_: int, c_bc_: float, nx_: int, nz_: int,
        ):  # fmt: skip
            for i, k in ti.ndrange((0, nx_), (0, nz_)):
                if solid_[i, idx_, k] == 0:
                    for s in ti.static(range(7)):
                        G_[i, idx_, k][s] = w_[s] * c_bc_

        @ti.kernel
        def bc_z(
            solid_: ti.template(), G_: ti.template(), w_: ti.template(),
            idx_: int, c_bc_: float, nx_: int, ny_: int,
        ):  # fmt: skip
            for i, j in ti.ndrange((0, nx_), (0, ny_)):
                if solid_[i, j, idx_] == 0:
                    for s in ti.static(range(7)):
                        G_[i, j, idx_][s] = w_[s] * c_bc_

        self._finalize_collide = finalize_collide
        self._stream = stream
        self._bc_x = bc_x
        self._bc_y = bc_y
        self._bc_z = bc_z
