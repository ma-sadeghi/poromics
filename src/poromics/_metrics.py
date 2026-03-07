import os

# from pathlib import Path
import numpy as np
from loguru import logger

from poromics import julia_helpers

__all__ = ["tortuosity_fd", "tortuosity_lbm", "DiffusionResult"]

# os.environ["PYTHON_JULIACALL_SYSIMAGE"] = str(Path(__file__).parents[2] / "sysimage.so")
os.environ["PYTHON_JULIACALL_STARTUP_FILE"] = "no"
os.environ["PYTHON_JULIACALL_AUTOLOAD_IPYTHON_EXTENSION"] = "no"

_jl = None
_taujl = None


def _ensure_julia():
    """Lazily initializes Julia and loads Tortuosity.jl on first call."""
    global _jl, _taujl
    if _jl is None:
        julia_helpers.ensure_julia_deps_ready(quiet=False)
        _jl = julia_helpers.init_julia(quiet=False)
        _taujl = julia_helpers.import_backend(_jl)
    return _jl, _taujl


class Result:
    """Container for storing the results sof a tortuosity simulation."""

    def __init__(self, im, axis, tau, D_eff, c, D=None):
        """
        Initializes a tortuosity result object.

        Args:
            im (ndarray): The boolean image used in the simulation.
            axis (int or str): The axis along which tortuosity was calculated.
            tau (float): The tortuosity value.
            D_eff (float): The effective diffusivity value.
            c (ndarray): The concentration solution reshaped to the image shape.
            D (ndarray): The diffusivity field used in the simulation.

        """
        self.im = im
        self.axis = axis
        self.tau = tau
        self.D_eff = D_eff
        self.c = c
        self.D = D

    def __repr__(self):
        return f"Result(τ = {self.tau:.2f}, axis = {self.axis}, variable D = {self.D is not None})"


def tortuosity_fd(
    im,
    *,
    axis: int,
    D: np.ndarray = None,
    rtol: float = 1e-5,
    gpu: bool = False,
    verbose: bool = False,
) -> Result:
    """
    Performs a tortuosity simulation on the given image along the specified axis.

    The function removes non-percolating paths from the image before performing
    the tortuosity calculation.

    Args:
        im (ndarray): The input image.
        axis (int): The axis along which to compute tortuosity (0=x, 1=y, 2=z).
        D (ndarray): Diffusivity field. If None, a uniform diffusivity of 1.0 is assumed.
        rtol (float): Relative tolerance for the solver.
        gpu (bool): If True, use GPU for computation.
        verbose (bool): If True, print additional information during the solution process.

    Returns:
        result: An object containing the boolean image, axis, tortuosity, and
            concentration.

    Raises:
        RuntimeError: If no percolating paths are found along the specified axis.
    """
    jl, taujl = _ensure_julia()
    axis_jl = jl.Symbol(["x", "y", "z"][axis])
    eps0 = taujl.Imaginator.phase_fraction(im)
    im = np.array(taujl.Imaginator.trim_nonpercolating_paths(im, axis=axis_jl))
    if jl.sum(im) == 0:
        raise RuntimeError("No percolating paths along the given axis found in the image.")
    eps = taujl.Imaginator.phase_fraction(im)
    if eps[1] != eps0[1]:
        # Trim the diffusivity field as well
        if D is not None:
            D[~im] = 0.0
        logger.warning("The image has been trimmed to ensure percolation.")
    sim = taujl.TortuositySimulation(im, D=D, axis=axis_jl, gpu=gpu)
    sol = taujl.solve(sim.prob, taujl.KrylovJL_CG(), verbose=verbose, reltol=rtol)
    c = taujl.vec_to_grid(sol.u, im)
    tau = taujl.tortuosity(c, axis=axis_jl, D=D)
    D_eff = taujl.effective_diffusivity(c, axis=axis_jl, D=D)
    return Result(np.asarray(im), axis, tau, D_eff, np.asarray(c), D)


class DiffusionResult:
    """Container for LBM diffusion simulation results."""

    def __init__(self, im, axis, tau, D_eff, porosity, formation_factor, c):
        """
        Parameters
        ----------
        im : ndarray
            The boolean image used in the simulation.
        axis : int
            The axis along which diffusion was computed.
        tau : float
            Tortuosity factor (always >= 1).
        D_eff : float
            Normalized effective diffusivity (D_eff / D_0).
        porosity : float
            Pore volume fraction.
        formation_factor : float
            Formation factor F = 1 / D_eff_norm.
        c : ndarray
            Steady-state concentration field.
        """
        self.im = im
        self.axis = axis
        self.tau = tau
        self.D_eff = D_eff
        self.porosity = porosity
        self.formation_factor = formation_factor
        self.c = c

    def __repr__(self):
        return (
            f"DiffusionResult(tau={self.tau:.4f}, D_eff={self.D_eff:.6f}, axis={self.axis})"
        )


def tortuosity_lbm(
    im,
    *,
    axis: int,
    D: float = 0.25,
    tol: float = 1e-2,
    n_steps: int = 100_000,
    sparse: bool = False,
) -> DiffusionResult:
    """Compute tortuosity and effective diffusivity using LBM (D3Q7 BGK).

    Solves the steady-state diffusion equation on the pore space of a
    3D binary image using the Lattice Boltzmann Method with a D3Q7 BGK
    collision operator.

    Parameters
    ----------
    im : ndarray, shape (nx, ny, nz)
        Binary image. True (or 1) = pore, False (or 0) = solid.
    axis : int
        Axis along which to apply the concentration gradient
        (0=x, 1=y, 2=z).
    D : float
        Bulk diffusivity in lattice units. Default 0.25.
        Relaxation time tau_D = 4*D + 0.5. D=0.25 gives tau_D=1.5,
        a good balance between speed and accuracy.
    tol : float
        Convergence tolerance on relative concentration change.
        Default 1e-2.
    n_steps : int
        Maximum number of LBM iterations. Default 100000.
    sparse : bool
        If True, use Taichi sparse storage to reduce memory on
        high-solid-fraction images. Default False.

    Returns
    -------
    result : DiffusionResult
        Contains tau, D_eff, porosity, formation_factor, and the
        steady-state concentration field.

    Raises
    ------
    RuntimeError
        If the image has no pore voxels.
    """
    from ._lbm._diffusion_solver import solve_diffusion

    if axis not in (0, 1, 2):
        raise ValueError(f"axis must be 0, 1, or 2, got {axis}")
    solid = (im == 0).astype(np.int8)
    if solid.sum() == solid.size:
        raise RuntimeError("Image has no pore voxels.")

    c, J_mean = solve_diffusion(
        solid,
        axis=axis,
        D=D,
        n_steps=n_steps,
        tol=tol,
        sparse=sparse,
    )

    # Compute transport properties from Fick's law
    pore_mask = solid == 0
    porosity = float(pore_mask.sum()) / pore_mask.size
    L = im.shape[axis]
    D_eff_lu = J_mean * L  # delta_c = 1.0
    D_eff_norm = D_eff_lu / D
    if D_eff_norm > 0:
        formation_factor = 1.0 / D_eff_norm
    else:
        formation_factor = float("inf")
    tau = formation_factor * porosity
    return DiffusionResult(
        im=np.asarray(im, dtype=bool),
        axis=axis,
        tau=tau,
        D_eff=D_eff_norm,
        porosity=porosity,
        formation_factor=formation_factor,
        c=c,
    )
