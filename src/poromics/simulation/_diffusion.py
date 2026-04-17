# Transient diffusion solver on 3D voxel images.
import time

import numpy as np
from loguru import logger

from ._progress import make_progress, update_progress
from .._lbm._lattice import axis_to_face, D3Q7Params

_d3q7 = D3Q7Params()


class TransientDiffusion:
    """Transient diffusion solver on 3D voxel images.

    Solves the time-dependent diffusion equation on the pore space
    using the Lattice Boltzmann Method (D3Q7 BGK). Boundary conditions
    are Dirichlet (fixed concentration) on the inlet/outlet faces and
    periodic on the remaining faces.

    Parameters
    ----------
    im : ndarray, shape (nx, ny, nz)
        Binary image. True (or 1) = pore, False (or 0) = solid.
    axis : int
        Axis along which to apply the concentration gradient
        (0=x, 1=y, 2=z).
    D : float
        Bulk diffusivity in m²/s.
    voxel_size : float
        Physical voxel edge length in metres.
    c_in : float
        Inlet concentration. Default 1.0.
    c_out : float
        Outlet concentration. Default 0.0.
    sparse : bool
        Use Taichi sparse storage. Default False.

    Examples
    --------
    >>> solver = TransientDiffusion(im, axis=0, D=1e-9, voxel_size=1e-6)
    >>> solver.run(tol=1e-2)
    >>> c = solver.concentration
    """

    def __init__(self, im, axis, D, voxel_size, c_in=1.0, c_out=0.0,sparse=False):  # fmt: skip
        if axis not in (0, 1, 2):
            raise ValueError(f"axis must be 0, 1, or 2, got {axis}")
        if D <= 0:
            raise ValueError(f"D must be > 0, got {D}")
        if voxel_size <= 0:
            raise ValueError(f"voxel_size must be > 0, got {voxel_size}")
        im_arr = np.atleast_3d(np.asarray(im))
        solid = (im_arr == 0).astype(np.int8)
        if solid.sum() == solid.size:
            raise RuntimeError("Image has no pore voxels.")

        self._axis = axis
        self._D = D
        self._voxel_size = float(voxel_size)
        self._dt = _d3q7.D * voxel_size**2 / D
        self._n_iterations = 0
        self._converged = False

        from .._lbm._taichi_helpers import ensure_taichi
        from .._lbm._diffusion_solver import _D3Q7Solver

        ti = ensure_taichi()
        self._solver = _D3Q7Solver(ti, solid, D=_d3q7.D, sparse=sparse)
        inlet, outlet = axis_to_face[axis]
        self._solver.set_bc(inlet, c_in)
        self._solver.set_bc(outlet, c_out)
        self._solver.init_fields()

    # ── Execution ─────────────────────────────────────────────────────

    def run(self, n_steps=100_000, tol=1e-3, log_every=500,verbose=False):  # fmt: skip
        """Run the solver to steady state.

        Parameters
        ----------
        n_steps : int
            Maximum number of iterations.
        tol : float or None
            Convergence tolerance on relative concentration change.
            None disables early stopping.
        log_every : int
            Log convergence every this many steps.
        verbose : bool
            Show a progress bar. Default False.
        """
        self._converged = False
        t_start = time.time()
        c_prev = self._solver.get_concentration()
        pbar = None
        if verbose:
            pbar = make_progress(n_steps, tol, "Diffusion")
        try:
            for step in range(n_steps + 1):
                self._solver.step()
                self._n_iterations += 1
                if step % log_every != 0:
                    continue
                elapsed = time.time() - t_start
                c_now = self._solver.get_concentration()
                c_total = np.sum(np.abs(c_now))
                c_change = np.sum(np.abs(c_now - c_prev))
                ratio = c_change / c_total if c_total > 0 else 0.0
                logger.info(
                    f"Step {step:>6d}/{n_steps}  "
                    f"|c|={c_total:.4e}  "
                    f"delta={ratio:.2e}  "
                    f"elapsed={elapsed:.1f}s"
                )
                if pbar is not None:
                    update_progress(pbar, step, ratio, tol, n_steps)
                if tol is not None and step > 0 and c_total > 0 and ratio < tol:
                    logger.info(
                        f"Converged at step {step} (delta|c|/|c|={ratio:.2e} < tol={tol:.2e})"
                    )
                    if pbar is not None:
                        pbar.n = 100
                        pbar.set_postfix_str("converged")
                        pbar.refresh()
                    self._converged = True
                    return
                c_prev = c_now
        finally:
            if pbar is not None:
                pbar.close()

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
    def concentration(self):
        """Concentration field, shape (nx, ny, nz).

        Units match ``c_in`` and ``c_out`` (default: dimensionless).
        """
        return self._solver.get_concentration()

    def flux(self, axis=None, n_slices=5):
        """Mean diffusive flux averaged across interior slices (lattice units).

        Uses Fick's law on the concentration gradient rather than
        distribution moments (which are unreliable at Dirichlet faces).
        Averages ``-D_lu * dc/dx_lu`` across ``n_slices`` slices evenly
        spaced in the central 60% of the domain, which smooths out
        local noise while avoiding boundary-layer artifacts. Equals
        the normalized effective diffusivity D_eff/D_0 for a unit
        concentration drop across the domain.

        Parameters
        ----------
        axis : int or None
            Axis along which to compute flux. Defaults to the
            axis used in the constructor.
        n_slices : int
            Number of interior slices to average. Default 5.

        Returns
        -------
        J_mean : float
        """
        if axis is None:
            axis = self._axis
        return self._solver.compute_flux(axis, n_slices=n_slices)
