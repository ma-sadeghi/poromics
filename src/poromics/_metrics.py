import os

# from pathlib import Path
import numpy as np
from loguru import logger

from poromics import julia_helpers

__all__ = ["tortuosity_fd"]

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
