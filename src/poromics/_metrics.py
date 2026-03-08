import atexit
import os
import pickle
import subprocess
import sys
import tempfile

import numpy as np

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

_JULIA_SUBPROCESS = os.environ.get("POROMICS_JULIA_SUBPROCESS", "1") == "1"

# ── In-process Julia (used when POROMICS_JULIA_SUBPROCESS=0) ─────────

os.environ["PYTHON_JULIACALL_STARTUP_FILE"] = "no"
os.environ["PYTHON_JULIACALL_AUTOLOAD_IPYTHON_EXTENSION"] = "no"

_jl = None
_taujl = None


def _ensure_julia():
    """Lazily initialize Julia in-process. Fails if Taichi is loaded."""
    global _jl, _taujl
    if _jl is not None:
        return _jl, _taujl
    from poromics._lbm._taichi_helpers import _ti
    if _ti is not None:
        raise RuntimeError(
            "Cannot initialize Julia in the same process as Taichi "
            "(LLVM symbol collision). Either call tortuosity_fd before "
            "any LBM function, or set POROMICS_JULIA_SUBPROCESS=1."
        )
    from poromics import julia_helpers
    julia_helpers.ensure_julia_deps_ready(quiet=False)
    _jl = julia_helpers.init_julia(quiet=False)
    _taujl = julia_helpers.import_backend(_jl)
    return _jl, _taujl


def _tortuosity_fd_inprocess(im, axis, D, rtol, gpu, verbose):
    """Run the Julia FD solver in the current process."""
    jl, taujl = _ensure_julia()
    axis_jl = jl.Symbol(["x", "y", "z"][axis])
    eps0 = taujl.Imaginator.phase_fraction(im)
    im = np.array(taujl.Imaginator.trim_nonpercolating_paths(im, axis=axis_jl))
    if jl.sum(im) == 0:
        raise RuntimeError(
            "No percolating paths along the given axis found in the image."
        )
    eps = taujl.Imaginator.phase_fraction(im)
    if eps[1] != eps0[1]:
        if D is not None:
            D[~im] = 0.0
    sim = taujl.TortuositySimulation(im, D=D, axis=axis_jl, gpu=gpu)
    sol = taujl.solve(
        sim.prob, taujl.KrylovJL_CG(), verbose=verbose, reltol=rtol
    )
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


# ── Persistent Julia worker (used when POROMICS_JULIA_SUBPROCESS=1) ──

_julia_proc = None
_julia_ctrl_w = None


def _get_julia_worker():
    """Return a running Julia worker subprocess, spawning one if needed.

    Uses a dedicated pipe fd for the control channel because Julia's
    runtime clobbers fd 0 (stdin) on initialization.
    """
    global _julia_proc, _julia_ctrl_w
    if _julia_proc is not None and _julia_proc.poll() is None:
        return _julia_proc, _julia_ctrl_w
    r_fd, w_fd = os.pipe()
    _julia_proc = subprocess.Popen(
        [sys.executable, "-m", "poromics._julia_worker", str(r_fd)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        pass_fds=(r_fd,),
    )
    os.close(r_fd)
    _julia_ctrl_w = os.fdopen(w_fd, "w")
    atexit.register(_shutdown_julia_worker)
    return _julia_proc, _julia_ctrl_w


def _shutdown_julia_worker():
    """Gracefully terminate the Julia worker on interpreter exit."""
    global _julia_proc, _julia_ctrl_w
    if _julia_ctrl_w is not None:
        try:
            _julia_ctrl_w.close()
        except OSError:
            pass
        _julia_ctrl_w = None
    if _julia_proc is not None and _julia_proc.poll() is None:
        _julia_proc.wait(timeout=5)
    _julia_proc = None


def _julia_call(payload):
    """Send a request to the persistent Julia worker and return the response."""
    proc, ctrl_w = _get_julia_worker()
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f_in, \
         tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f_out:
        in_path, out_path = f_in.name, f_out.name
        pickle.dump(payload, f_in)
    try:
        ctrl_w.write(f"{in_path}\t{out_path}\n")
        ctrl_w.flush()
        ack = proc.stdout.readline()
        if not ack:
            stderr = proc.stderr.read().decode(errors="replace")
            raise RuntimeError(f"Julia worker died:\n{stderr}")
        with open(out_path, "rb") as f:
            response = pickle.load(f)
    finally:
        os.unlink(in_path)
        os.unlink(out_path)
    if not response["ok"]:
        exc_cls = {"RuntimeError": RuntimeError, "ImportError": ImportError}
        cls = exc_cls.get(response["error_type"], RuntimeError)
        raise cls(response["error_msg"])
    return response["result"]


# ── Public API ────────────────────────────────────────────────────────


def tortuosity_fd(
    im,
    *,
    axis: int,
    D: np.ndarray = None,
    rtol: float = 1e-5,
    gpu: bool = False,
    verbose: bool = False,
) -> "TortuosityResult":
    """Compute tortuosity via Julia FD solver.

    By default (``POROMICS_JULIA_SUBPROCESS=1``), runs Julia in a
    persistent subprocess to avoid LLVM symbol collisions with Taichi.
    The worker stays alive so Julia's JIT cache is reused across calls.

    Set ``POROMICS_JULIA_SUBPROCESS=0`` to run Julia in-process (faster
    if Taichi is not used). An error is raised if Taichi was already
    initialized in the same process.

    Parameters
    ----------
    im : ndarray
        The input image.
    axis : int
        The axis along which to compute tortuosity (0=x, 1=y, 2=z).
    D : ndarray, optional
        Diffusivity field. If None, uniform diffusivity of 1.0 is assumed.
    rtol : float
        Relative tolerance for the solver.
    gpu : bool
        If True, use GPU for computation.
    verbose : bool
        If True, print additional information during the solution process.

    Returns
    -------
    result : TortuosityResult

    Raises
    ------
    RuntimeError
        If no percolating paths are found along the specified axis.
    """
    if not _JULIA_SUBPROCESS:
        return _tortuosity_fd_inprocess(im, axis, D, rtol, gpu, verbose)
    payload = {
        "im": np.asarray(im),
        "axis": axis,
        "D": D,
        "rtol": rtol,
        "gpu": gpu,
        "verbose": verbose,
    }
    return TortuosityResult(**_julia_call(payload))


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
