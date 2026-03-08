import os

# from pathlib import Path
import numpy as np
from loguru import logger

from poromics import julia_helpers
from poromics._lbm._lattice import D3Q7Params, D3Q19Params
from poromics.simulation import TransientDiffusion, TransientFlow

_d3q7 = D3Q7Params()
_d3q19 = D3Q19Params()

__all__ = [
    "tortuosity_fd",
    "tortuosity_lbm",
    "permeability_lbm",
    "SimulationResult",
    "TortuosityResult",
    "PermeabilityResult",
]

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


def tortuosity_fd(
    im,
    *,
    axis: int,
    D: np.ndarray = None,
    rtol: float = 1e-5,
    gpu: bool = False,
    verbose: bool = False,
) -> "TortuosityResult":
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
    pore_mask = np.asarray(im, dtype=bool)
    porosity = float(pore_mask.sum()) / pore_mask.size
    formation_factor = 1.0 / D_eff if D_eff > 0 else float("inf")
    return TortuosityResult(
        im=np.asarray(im, dtype=bool),
        axis=axis,
        porosity=porosity,
        tau=tau,
        D_eff=D_eff,
        c=np.asarray(c),
        formation_factor=formation_factor,
        D=D,
    )


# ── Result classes ────────────────────────────────────────────────────


class SimulationResult:
    """Base class for simulation results.

    Parameters
    ----------
    im : ndarray
        The boolean image used in the simulation.
    axis : int
        The axis along which the simulation was run.
    porosity : float
        Pore volume fraction.
    """

    def __init__(self, im, axis, porosity):
        self.im = im
        self.axis = axis
        self.porosity = porosity


class TortuosityResult(SimulationResult):
    """Results from a tortuosity / effective diffusivity simulation.

    Parameters
    ----------
    im : ndarray
        The boolean image used in the simulation.
    axis : int
        The axis along which diffusion was computed.
    porosity : float
        Pore volume fraction.
    tau : float
        Tortuosity factor (>= 1).
    D_eff : float
        Normalized effective diffusivity (D_eff / D_0).
    c : ndarray
        Steady-state concentration field.
    formation_factor : float
        Formation factor F = 1 / D_eff_norm.
    D : float or ndarray
        Bulk diffusivity (float for uniform, ndarray for spatially
        variable).
    """

    def __init__(self, im, axis, porosity, tau, D_eff, c,
                 formation_factor=None, D=None):  # fmt: skip
        super().__init__(im, axis, porosity)
        self.tau = tau
        self.D_eff = D_eff
        self.c = c
        self.formation_factor = formation_factor
        self.D = D

    def __repr__(self):
        return (
            f"TortuosityResult(tau={self.tau:.4f}, D_eff={self.D_eff:.6f}, axis={self.axis})"
        )


_MD_CONVERSION = 9.869233e-16  # 1 milliDarcy in m²


class PermeabilityResult(SimulationResult):
    """Results from an LBM flow / permeability simulation.

    Parameters
    ----------
    im : ndarray
        The boolean image used in the simulation.
    axis : int
        The axis along which flow was computed.
    porosity : float
        Pore volume fraction.
    k_lu : float
        Permeability in lattice units (voxels²).
    k_m2 : float
        Permeability in m².
    k_mD : float
        Permeability in milliDarcy.
    u_darcy : float
        Darcy (superficial) velocity in m/s.
    u_pore : float
        Mean pore-space velocity in m/s.
    velocity : ndarray, shape (nx, ny, nz, 3)
        Steady-state velocity field in m/s.
    """

    def __init__(self, im, axis, porosity, k_lu, k_m2, k_mD,
                 u_darcy, u_pore, velocity):  # fmt: skip
        super().__init__(im, axis, porosity)
        self.k_lu = k_lu
        self.k_m2 = k_m2
        self.k_mD = k_mD
        self.u_darcy = u_darcy
        self.u_pore = u_pore
        self.velocity = velocity

    def __repr__(self):
        return (
            f"PermeabilityResult(k_m2={self.k_m2:.6e}, k_mD={self.k_mD:.4f}, "
            f"axis={self.axis})"
        )


# ── LBM metric functions ─────────────────────────────────────────────


def tortuosity_lbm(
    im,
    *,
    axis: int,
    D: float = 1e-9,
    voxel_size: float,
    tol: float = 1e-2,
    n_steps: int = 100_000,
    sparse: bool = False,
) -> TortuosityResult:
    """Compute tortuosity and effective diffusivity using LBM (D3Q7 BGK).

    Solves the steady-state diffusion equation on the pore space of a
    3D binary image using the Lattice Boltzmann Method.

    Parameters
    ----------
    im : ndarray, shape (nx, ny, nz)
        Binary image. True (or 1) = pore, False (or 0) = solid.
    axis : int
        Axis along which to apply the concentration gradient
        (0=x, 1=y, 2=z).
    D : float
        Bulk diffusivity in m²/s. Default 1e-9.
    voxel_size : float
        Physical voxel edge length in metres.
    tol : float
        Convergence tolerance on relative concentration change.
    n_steps : int
        Maximum number of LBM iterations.
    sparse : bool
        Use Taichi sparse storage.

    Returns
    -------
    result : TortuosityResult
    """
    solver = TransientDiffusion(im, axis=axis, D=D, voxel_size=voxel_size, sparse=sparse)
    solver.run(n_steps=n_steps, tol=tol)
    c = solver.concentration

    pore_mask = np.asarray(im, dtype=bool)
    porosity = float(pore_mask.sum()) / pore_mask.size
    L = im.shape[axis]
    J_mean = solver.flux(axis)
    D_eff_lu = J_mean * L  # delta_c = 1.0
    D_eff_norm = D_eff_lu / _d3q7.D_lu
    if D_eff_norm > 0:
        formation_factor = 1.0 / D_eff_norm
    else:
        formation_factor = float("inf")
    tau = formation_factor * porosity

    return TortuosityResult(
        im=np.asarray(im, dtype=bool),
        axis=axis,
        porosity=porosity,
        tau=tau,
        D_eff=D_eff_norm,
        formation_factor=formation_factor,
        c=c,
        D=D,
    )


def permeability_lbm(
    im,
    *,
    axis: int,
    nu: float = 1e-6,
    voxel_size: float,
    tol: float = 1e-3,
    n_steps: int = 100_000,
    sparse: bool = False,
) -> PermeabilityResult:
    """Compute absolute permeability using LBM (D3Q19 MRT).

    Solves creeping (Stokes) flow on the pore space of a 3D binary image
    using the Lattice Boltzmann Method. Permeability is extracted via
    Darcy's law.

    Parameters
    ----------
    im : ndarray, shape (nx, ny, nz)
        Binary image. True (or 1) = pore, False (or 0) = solid.
    axis : int
        Axis along which to apply the pressure gradient
        (0=x, 1=y, 2=z).
    nu : float
        Kinematic viscosity in m²/s. Default 1e-6 (water at ~20 °C).
    voxel_size : float
        Physical voxel edge length in metres.
    tol : float
        Convergence tolerance on relative velocity change.
    n_steps : int
        Maximum number of LBM iterations.
    sparse : bool
        Use Taichi sparse storage.

    Returns
    -------
    result : PermeabilityResult
    """
    solver = TransientFlow(im, axis=axis, nu=nu, voxel_size=voxel_size, sparse=sparse)
    solver.run(n_steps=n_steps, tol=tol)

    # Work in lattice units for Darcy's law, then convert
    v_lu = solver._solver.get_velocity()
    solid = (np.asarray(im) == 0).astype(np.int8)
    pore_mask = solid == 0
    porosity = float(pore_mask.sum()) / pore_mask.size
    L = im.shape[axis]

    v_flow_lu = v_lu[..., axis]
    u_darcy_lu = float(np.mean(v_flow_lu))
    u_pore_lu = float(np.mean(v_flow_lu[pore_mask]))
    grad_P_lu = (solver._rho_in - solver._rho_out) * _d3q19.cs2 / L
    k_lu = u_darcy_lu * _d3q19.nu_lu / grad_P_lu

    # Convert to physical units
    dx = voxel_size
    k_m2 = k_lu * dx ** 2
    k_mD = k_m2 / _MD_CONVERSION
    lu_to_phys = dx / solver.dt
    u_darcy = u_darcy_lu * lu_to_phys
    u_pore = u_pore_lu * lu_to_phys

    return PermeabilityResult(
        im=np.asarray(im, dtype=bool),
        axis=axis,
        porosity=porosity,
        k_lu=k_lu,
        k_m2=k_m2,
        k_mD=k_mD,
        u_darcy=u_darcy,
        u_pore=u_pore,
        velocity=solver.velocity,
    )
