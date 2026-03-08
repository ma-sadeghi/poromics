# Transient single-phase flow solver on 3D voxel images.
import time

import numpy as np
from loguru import logger

from ._progress import make_progress, update_progress
from .._lbm._lattice import axis_to_face, D3Q19Params

_d3q19 = D3Q19Params()


class TransientFlow:
    """Transient single-phase flow solver on 3D voxel images.

    Solves the incompressible Navier-Stokes equations (Stokes regime)
    on the pore space using the Lattice Boltzmann Method (D3Q19 MRT).
    Boundary conditions are fixed-pressure on the inlet/outlet faces
    and periodic on the remaining faces.

    Parameters
    ----------
    im : ndarray, shape (nx, ny, nz)
        Binary image. True (or 1) = pore, False (or 0) = solid.
    axis : int
        Axis along which to apply the pressure gradient
        (0=x, 1=y, 2=z).
    nu : float
        Kinematic viscosity in m²/s.
    voxel_size : float
        Physical voxel edge length in metres.
    rho_in : float
        Inlet density (pressure BC). Default 1.0.
    rho_out : float
        Outlet density (pressure BC). Default 0.99.
    sparse : bool
        Use Taichi sparse storage. Default False.

    Examples
    --------
    >>> solver = TransientFlow(im, axis=0, nu=1e-6, voxel_size=1e-6)
    >>> solver.run(tol=1e-3)
    >>> v = solver.velocity    # m/s
    >>> P = solver.pressure    # Pa (gauge)
    """

    def __init__(self, im, axis, nu, voxel_size, rho_in=1.0, rho_out=0.99,
                 sparse=False):  # fmt: skip
        if axis not in (0, 1, 2):
            raise ValueError(f"axis must be 0, 1, or 2, got {axis}")
        if nu <= 0:
            raise ValueError(f"nu must be > 0, got {nu}")
        if voxel_size <= 0:
            raise ValueError(f"voxel_size must be > 0, got {voxel_size}")
        im_arr = np.atleast_3d(np.asarray(im))
        solid = (im_arr == 0).astype(np.int8)
        if solid.sum() == solid.size:
            raise RuntimeError("Image has no pore voxels.")

        self._axis = axis
        self._nu = nu
        self._voxel_size = float(voxel_size)
        self._rho_in = rho_in
        self._rho_out = rho_out
        self._dt = _d3q19.nu_lu * voxel_size**2 / nu
        self._n_iterations = 0
        self._converged = False

        from .._lbm._taichi_helpers import ensure_taichi
        from .._lbm._flow_solver import _D3Q19Solver

        ti = ensure_taichi()
        self._solver = _D3Q19Solver(ti, solid, nu=_d3q19.nu_lu, sparse=sparse)
        inlet, outlet = axis_to_face[axis]
        self._solver.set_bc_rho(inlet, rho_in)
        self._solver.set_bc_rho(outlet, rho_out)
        self._solver.init_fields()

    # ── Execution ─────────────────────────────────────────────────────

    def run(self, n_steps=100_000, tol=1e-3, log_every=500,
            verbose=False):  # fmt: skip
        """Run the solver to steady state.

        Parameters
        ----------
        n_steps : int
            Maximum number of iterations.
        tol : float or None
            Convergence tolerance on relative velocity change.
            None disables early stopping.
        log_every : int
            Log convergence every this many steps.
        verbose : bool
            Show a rich progress bar. Default False.
        """
        self._converged = False
        t_start = time.time()
        v_prev = self._solver.get_velocity()
        progress, task = None, None
        if verbose:
            progress, task = make_progress(n_steps, tol, "Flow")
            progress.start()
        try:
            for step in range(n_steps + 1):
                self._solver.step()
                self._n_iterations += 1
                if step % log_every != 0:
                    continue
                elapsed = time.time() - t_start
                v_now = self._solver.get_velocity()
                v_total = np.sum(np.abs(v_now))
                v_change = np.sum(np.abs(v_now - v_prev))
                ratio = v_change / v_total if v_total > 0 else 0.0
                logger.info(
                    f"Step {step:>6d}/{n_steps}  "
                    f"|v|={v_total:.4e}  "
                    f"delta={ratio:.2e}  "
                    f"elapsed={elapsed:.1f}s"
                )
                if progress is not None:
                    update_progress(progress, task, step, ratio, tol, n_steps)
                if tol is not None and v_total > 0 and ratio < tol:
                    logger.info(
                        f"Converged at step {step} "
                        f"(delta|v|/|v|={ratio:.2e} < tol={tol:.2e})"
                    )
                    if progress is not None:
                        progress.update(task, completed=100, status="converged")
                    self._converged = True
                    return
                v_prev = v_now
        finally:
            if progress is not None:
                progress.stop()

    def step(self):
        """Advance by one time step."""
        self._solver.step()
        self._n_iterations += 1

    # ── Properties ────────────────────────────────────────────────────

    @property
    def converged(self):
        """Whether the solver has converged (set by ``run()``)."""
        return self._converged

    @property
    def n_iterations(self):
        """Total number of time steps taken."""
        return self._n_iterations

    @property
    def voxel_size(self):
        """Voxel edge length in metres."""
        return self._voxel_size

    @property
    def dt(self):
        """Physical time step in seconds."""
        return self._dt

    @property
    def velocity(self):
        """Velocity field in m/s, shape (nx, ny, nz, 3)."""
        v_lu = self._solver.get_velocity()
        return v_lu * (self._voxel_size / self._dt)

    @property
    def pressure(self):
        """Gauge pressure field in Pa, shape (nx, ny, nz).

        Pressure is relative to the outlet face density. Uses the LBM
        equation of state ``p = rho * cs²`` with the lattice reference
        density ``rho_0 = 1``, so no additional fluid density parameter
        is needed.
        """
        rho_lu = self._solver.get_density()
        scale = _d3q19.cs2 * (self._voxel_size / self._dt) ** 2
        return (rho_lu - self._rho_out) * scale
