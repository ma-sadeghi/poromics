import atexit
import os
import pickle
import subprocess
import sys
import tempfile

import numpy as np
from loguru import logger

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


def _trim_nonpercolating(im, axis):
    """Remove pore voxels that don't percolate between inlet/outlet faces.

    Uses PoreSpy to identify and remove dead-end regions that don't
    contribute to transport. Returns the trimmed boolean image.
    """
    import porespy as ps

    inlets = ps.generators.faces(im.shape, inlet=axis)
    outlets = ps.generators.faces(im.shape, outlet=axis)
    return ps.filters.trim_nonpercolating_paths(im, inlets=inlets, outlets=outlets)

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
    sim = taujl.TortuositySimulation(im, D=D, axis=axis_jl, gpu=gpu)
    sol = taujl.solve(sim.prob, taujl.KrylovJL_CG(), verbose=verbose, reltol=rtol)
    c = taujl.vec_to_grid(sol.u, im)
    tau = taujl.tortuosity(c, axis=axis_jl, D=D)
    D_eff = taujl.effective_diffusivity(c, axis=axis_jl, D=D)
    porosity = float(im.sum()) / im.size
    formation_factor = 1.0 / D_eff if D_eff > 0 else float("inf")
    return TortuosityResult(
        im=im,
        axis=axis,
        porosity=porosity,
        tau=tau,
        D_eff=D_eff,
        c=np.asarray(c),
        formation_factor=formation_factor,
        D=D,
        converged=True,
        n_iterations=None,
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
    with (
        tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f_in,
        tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f_out,
    ):
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
    D: np.ndarray | None = None,
    rtol: float = 1e-5,
    gpu: bool = False,
    verbose: bool = True,
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
    im = np.atleast_3d(np.asarray(im, dtype=bool))
    n_pore_before = int(im.sum())
    im = _trim_nonpercolating(im, axis)
    if im.sum() == 0:
        raise RuntimeError("No percolating paths along the given axis found in the image.")
    n_removed = n_pore_before - int(im.sum())
    if n_removed > 0:
        logger.warning(f"Trimmed {n_removed} non-percolating pore voxels from the image.")
        if D is not None:
            D = np.array(D, copy=True)
            D[~im] = 0.0
    if not _JULIA_SUBPROCESS:
        return _tortuosity_fd_inprocess(im, axis, D, rtol, gpu, verbose)
    payload = {
        "im": im,
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

    Tortuosity, effective diffusivity, and the concentration field are
    all dimensionless (or normalized), so no ``rescale`` method is
    provided. See ``PermeabilityResult.rescale`` for the flow analogue.

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
    converged : bool
        Whether the solver reached the requested tolerance. False
        means the reported tau / D_eff are from a pre-steady-state
        field and should not be trusted without further iteration.
    n_iterations : int or None
        Iterations the solver took. None for non-iterative backends.
    """

    def __init__(self, im, axis, porosity, tau, D_eff, c,
                 formation_factor=None, D=None, *,
                 converged=True, n_iterations=None):  # fmt: skip
        super().__init__(im, axis, porosity)
        self.tau = tau
        self.D_eff = D_eff
        self.c = c
        self.formation_factor = formation_factor
        self.D = D
        self.converged = converged
        self.n_iterations = n_iterations

    def __repr__(self):
        lines = [
            "TortuosityResult:",
            f"  axis       = {self.axis}",
            f"  porosity   = {self.porosity:.4f}",
            f"  tau        = {self.tau:.4f}",
            f"  D_eff/D    = {self.D_eff:.6f}",
            f"  F          = {self.formation_factor:.4f}",
            f"  converged  = {self.converged}" + (
                f" ({self.n_iterations} iters)"
                if self.n_iterations is not None else ""
            ),
        ]
        return "\n".join(lines)

    def plot_concentration(self, z=None, ax=None):
        """Plot concentration field on a 2D slice.

        Parameters
        ----------
        z : int or None
            Slice index along the third axis. Defaults to nz // 2.
        ax : matplotlib.axes.Axes or None
            If provided, plot concentration only on this axes.
            If None, create a side-by-side figure (pore structure +
            concentration).

        Returns
        -------
        fig : matplotlib.figure.Figure
        axes : ndarray of Axes (if ax is None) or Axes
        """
        import matplotlib.pyplot as plt

        im = np.atleast_3d(self.im)
        c = np.atleast_3d(self.c)
        if z is None:
            z = im.shape[2] // 2

        c_slice = c[:, :, z].astype(float)
        c_slice[~im[:, :, z]] = np.nan

        import matplotlib.ticker as ticker

        if ax is not None:
            mappable = ax.imshow(c_slice, cmap="viridis", interpolation="nearest")
            ax.set_title("Concentration field")
            cb = ax.figure.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04, label="c")
            cb.ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
            cb.ax.ticklabel_format(style="scientific", scilimits=(-1, 1))
            return ax.figure, ax

        from mpl_toolkits.axes_grid1 import make_axes_locatable

        fig, axes = plt.subplots(1, 2, figsize=(7.5, 4))
        axes[0].imshow(im[:, :, z], cmap="gray", interpolation="nearest")
        axes[0].set_title("Pore structure")
        div0 = make_axes_locatable(axes[0])
        phantom = div0.append_axes("right", size="5%", pad=0.05)
        phantom.set_visible(False)

        mappable = axes[1].imshow(c_slice, cmap="viridis", interpolation="nearest")
        axes[1].set_title("Concentration field")
        div1 = make_axes_locatable(axes[1])
        cax1 = div1.append_axes("right", size="5%", pad=0.05)
        cb = fig.colorbar(mappable, cax=cax1, label="c")
        cb.ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        cb.ax.ticklabel_format(style="scientific", scilimits=(-1, 1))
        fig.tight_layout()
        return fig, axes


_MD_CONVERSION = 9.869233e-16  # 1 milliDarcy in m²


class PermeabilityResult(SimulationResult):
    """Results from an LBM flow / permeability simulation.

    Use ``rescale`` to reinterpret the simulation for a different
    voxel size, fluid viscosity, or fluid density without rerunning
    the solver.

    Parameters
    ----------
    im : ndarray
        The boolean image used in the simulation.
    axis : int
        The axis along which flow was computed.
    porosity : float
        Pore volume fraction.
    k : float
        Permeability in m².
    u_darcy : float
        Darcy (superficial) velocity in m/s.
    u_pore : float
        Mean pore-space velocity in m/s.
    velocity : ndarray, shape (nx, ny, nz, 3)
        Steady-state velocity field in m/s.
    kinematic_pressure : ndarray, shape (nx, ny, nz)
        Gauge kinematic pressure (p / rho) in m²/s². Density-free;
        always populated.
    pressure : ndarray or None, shape (nx, ny, nz)
        Gauge pressure in Pa. Populated only when a fluid density was
        provided (``rho`` in ``permeability_lbm`` or ``rescale``).
    converged : bool
        Whether the solver reached the requested tolerance. False
        means the reported k / velocity are from a pre-steady-state
        field and should not be trusted without further iteration.
    n_iterations : int or None
        Iterations the solver took. None for non-iterative backends.
    """

    def __init__(self, im, axis, porosity, k, u_darcy, u_pore,
                 velocity, kinematic_pressure, pressure=None, *,
                 converged=True, n_iterations=None,
                 _velocity_lu=None, _rho_lu=None, _rho_out=None,
                 _k_lu=None, _u_darcy_lu=None,
                 _u_pore_lu=None):  # fmt: skip
        super().__init__(im, axis, porosity)
        self.k = k
        self.u_darcy = u_darcy
        self.u_pore = u_pore
        self.velocity = velocity
        self.kinematic_pressure = kinematic_pressure
        self.pressure = pressure
        self.converged = converged
        self.n_iterations = n_iterations
        self._velocity_lu = _velocity_lu
        self._rho_lu = _rho_lu
        self._rho_out = _rho_out
        self._k_lu = _k_lu
        self._u_darcy_lu = _u_darcy_lu
        self._u_pore_lu = _u_pore_lu

    def rescale(self, voxel_size, nu, rho=None):
        """Reinterpret the simulation for different physical parameters.

        Returns a new ``PermeabilityResult`` with recomputed physical
        quantities. The underlying lattice-unit fields are unchanged,
        so no solver re-run is needed. This is valid because
        permeability depends only on geometry, and the Stokes equations
        are linear.

        Parameters
        ----------
        voxel_size : float
            Physical voxel edge length in metres.
        nu : float
            Kinematic viscosity in m²/s.
        rho : float or None
            Fluid density in kg/m³. Required for pressure conversion
            to Pa. If None, the ``pressure`` field is omitted (set to
            None); ``kinematic_pressure`` is always populated.

        Returns
        -------
        result : PermeabilityResult
        """
        if self._velocity_lu is None:
            raise RuntimeError(
                "Lattice-unit data not available for rescaling."
            )
        dt = _d3q19.nu * voxel_size**2 / nu
        lu_to_phys = voxel_size / dt
        k = self._k_lu * voxel_size**2
        u_darcy = self._u_darcy_lu * lu_to_phys
        u_pore = self._u_pore_lu * lu_to_phys
        velocity = self._velocity_lu * lu_to_phys
        gauge_rho = self._rho_lu - self._rho_out
        kinematic_pressure = _d3q19.cs2 * lu_to_phys**2 * gauge_rho
        pressure = rho * kinematic_pressure if rho is not None else None
        return PermeabilityResult(
            im=self.im, axis=self.axis, porosity=self.porosity,
            k=k, u_darcy=u_darcy, u_pore=u_pore,
            velocity=velocity, kinematic_pressure=kinematic_pressure,
            pressure=pressure, converged=self.converged,
            n_iterations=self.n_iterations,
            _velocity_lu=self._velocity_lu, _rho_lu=self._rho_lu,
            _rho_out=self._rho_out, _k_lu=self._k_lu,
            _u_darcy_lu=self._u_darcy_lu,
            _u_pore_lu=self._u_pore_lu,
        )

    def __repr__(self):
        k_mD = self.k / _MD_CONVERSION
        lines = [
            "PermeabilityResult:",
            f"  axis       = {self.axis}",
            f"  porosity   = {self.porosity:.4f}",
            f"  k          = {self.k:.4e} m\u00b2 ({k_mD:.2f} mD)",
            f"  u_darcy    = {self.u_darcy:.4e} m/s",
            f"  u_pore     = {self.u_pore:.4e} m/s",
            f"  converged  = {self.converged}" + (
                f" ({self.n_iterations} iters)"
                if self.n_iterations is not None else ""
            ),
        ]
        return "\n".join(lines)


    def plot_velocity(self, z=None, ax=None):
        """Plot velocity magnitude and streamlines on a 2D slice.

        Parameters
        ----------
        z : int or None
            Slice index along the third axis. Defaults to nz // 2.
        ax : matplotlib.axes.Axes or None
            If provided, plot velocity magnitude only on this axes.
            If None, create a side-by-side figure (magnitude +
            streamlines).

        Returns
        -------
        fig : matplotlib.figure.Figure
        axes : ndarray of Axes (if ax is None) or Axes
        """
        import matplotlib.pyplot as plt

        im = np.atleast_3d(self.im)
        vel = np.atleast_3d(self.velocity)
        if z is None:
            z = im.shape[2] // 2
        v = vel[:, :, z, :]
        speed = np.linalg.norm(v, axis=-1)
        solid_mask = ~im[:, :, z]
        speed_masked = speed.copy()
        speed_masked[solid_mask] = np.nan

        import matplotlib.ticker as ticker

        if ax is not None:
            mappable = ax.imshow(speed_masked, cmap="viridis", interpolation="nearest")
            ax.set_title("Velocity magnitude")
            cb = ax.figure.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04, label="|u| (m/s)")
            cb.ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
            cb.ax.ticklabel_format(style="scientific", scilimits=(-1, 1))
            return ax.figure, ax

        from mpl_toolkits.axes_grid1 import make_axes_locatable

        fig, axes = plt.subplots(1, 2, figsize=(7.5, 4))
        mappable0 = axes[0].imshow(speed_masked, cmap="viridis", interpolation="nearest")
        axes[0].set_title("Velocity magnitude")
        div0 = make_axes_locatable(axes[0])
        cax0 = div0.append_axes("right", size="5%", pad=0.05)
        cb = fig.colorbar(mappable0, cax=cax0, label="|u| (m/s)")
        cb.ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        cb.ax.ticklabel_format(style="scientific", scilimits=(-1, 1))

        nx, ny = im.shape[0], im.shape[1]
        X, Y = np.meshgrid(np.arange(ny), np.arange(nx))
        lw = 3 * speed / speed.max() + 0.1 if speed.max() > 0 else np.ones_like(speed)
        axes[1].imshow(im[:, :, z], cmap="gray", interpolation="nearest", alpha=0.3)
        axes[1].streamplot(
            X, Y, v[:, :, 1], v[:, :, 0], color=speed,
            cmap="viridis", density=2, linewidth=lw, arrowstyle="fancy",
        )
        axes[1].set_xlim(-0.5, ny - 0.5)
        axes[1].set_ylim(nx - 0.5, -0.5)
        axes[1].set_title("Velocity streamlines")
        div1 = make_axes_locatable(axes[1])
        phantom = div1.append_axes("right", size="5%", pad=0.05)
        phantom.set_visible(False)
        fig.tight_layout()
        return fig, axes

    def plot_pressure(self, z=None, ax=None, kinematic=False):
        """Plot gauge pressure field on a 2D slice.

        Parameters
        ----------
        z : int or None
            Slice index along the third axis. Defaults to nz // 2.
        ax : matplotlib.axes.Axes or None
            If provided, plot on this axes. If None, create a new
            figure.
        kinematic : bool
            Plot ``kinematic_pressure`` (m²/s²) instead of
            ``pressure`` (Pa). Useful when no fluid density was
            provided.

        Returns
        -------
        fig : matplotlib.figure.Figure
        ax : matplotlib.axes.Axes
        """
        import matplotlib.pyplot as plt

        if kinematic:
            field, title, label = (
                self.kinematic_pressure,
                "Kinematic pressure (m²/s²)",
                "p/ρ (m²/s²)",
            )
        else:
            if self.pressure is None:
                raise RuntimeError(
                    "Pressure in Pa is unavailable because no fluid "
                    "density was provided. Pass rho=... to "
                    "permeability_lbm / rescale, or call with "
                    "kinematic=True."
                )
            field, title, label = self.pressure, "Pressure field (Pa)", "P (Pa)"

        im = np.atleast_3d(self.im)
        P = np.atleast_3d(field)
        if z is None:
            z = P.shape[2] // 2
        p_slice = P[:, :, z].astype(float)
        p_slice[~im[:, :, z]] = np.nan

        if ax is None:
            fig, ax = plt.subplots(figsize=(4, 4))
        else:
            fig = ax.figure
        import matplotlib.ticker as ticker
        mappable = ax.imshow(p_slice, cmap="coolwarm", interpolation="nearest")
        ax.set_title(title)
        cb = fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04, label=label)
        cb.ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        cb.ax.ticklabel_format(style="scientific", scilimits=(-1, 1))
        plt.tight_layout()
        return fig, ax


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
    verbose: bool = True,
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
    verbose : bool
        Show a rich progress bar. Default False.

    Returns
    -------
    result : TortuosityResult
    """
    im = np.atleast_3d(np.asarray(im, dtype=bool))
    n_pore_before = int(im.sum())
    im = _trim_nonpercolating(im, axis)
    if im.sum() == 0:
        raise RuntimeError("No percolating paths along the given axis found in the image.")
    n_removed = n_pore_before - int(im.sum())
    if n_removed > 0:
        logger.warning(f"Trimmed {n_removed} non-percolating pore voxels from the image.")
    solver = TransientDiffusion(im, axis=axis, D=D, voxel_size=voxel_size, sparse=sparse)
    solver.run(n_steps=n_steps, tol=tol, verbose=verbose)
    if not solver.converged:
        logger.warning(
            f"LBM diffusion solver did not converge after "
            f"{solver.n_iterations} steps (tol={tol:.2e}). "
            f"Reported tau / D_eff are from a pre-steady-state field; "
            f"increase n_steps or loosen tol."
        )
    c = solver.concentration

    porosity = float(im.sum()) / im.size
    L = im.shape[axis]
    J_mean = solver.flux(axis)
    D_eff_lu = J_mean * L  # delta_c = 1.0
    D_eff_norm = D_eff_lu / _d3q7.D
    if D_eff_norm > 0:
        formation_factor = 1.0 / D_eff_norm
    else:
        formation_factor = float("inf")
    tau = formation_factor * porosity

    return TortuosityResult(
        im=im,
        axis=axis,
        porosity=porosity,
        tau=tau,
        D_eff=D_eff_norm,
        formation_factor=formation_factor,
        c=c,
        D=D,
        converged=solver.converged,
        n_iterations=solver.n_iterations,
    )


def permeability_lbm(
    im,
    *,
    axis: int,
    nu: float = 1e-6,
    rho: float | None = None,
    voxel_size: float,
    tol: float = 1e-3,
    n_steps: int = 100_000,
    sparse: bool = False,
    verbose: bool = True,
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
    rho : float or None
        Fluid density in kg/m³. Required to return ``pressure`` in
        pascals; if None, ``pressure`` is set to None and only
        ``kinematic_pressure`` (m²/s²) is populated. Permeability and
        velocity are independent of density (Stokes regime).
    voxel_size : float
        Physical voxel edge length in metres.
    tol : float
        Convergence tolerance on relative velocity change.
    n_steps : int
        Maximum number of LBM iterations.
    sparse : bool
        Use Taichi sparse storage.
    verbose : bool
        Show a rich progress bar. Default False.

    Returns
    -------
    result : PermeabilityResult
    """
    im = np.atleast_3d(np.asarray(im, dtype=bool))
    n_pore_before = int(im.sum())
    im = _trim_nonpercolating(im, axis)
    if im.sum() == 0:
        raise RuntimeError("No percolating paths along the given axis found in the image.")
    n_removed = n_pore_before - int(im.sum())
    if n_removed > 0:
        logger.warning(f"Trimmed {n_removed} non-percolating pore voxels from the image.")
    solver = TransientFlow(im, axis=axis, nu=nu, rho=rho,
                           voxel_size=voxel_size, sparse=sparse)  # fmt: skip
    solver.run(n_steps=n_steps, tol=tol, verbose=verbose)
    if not solver.converged:
        logger.warning(
            f"LBM flow solver did not converge after "
            f"{solver.n_iterations} steps (tol={tol:.2e}). "
            f"Reported k / velocity are from a pre-steady-state field; "
            f"increase n_steps or loosen tol."
        )

    # Work in lattice units for Darcy's law, then convert
    v_lu = solver._solver.get_velocity()
    rho_lu = solver._solver.get_density()
    pore_mask = im
    porosity = float(pore_mask.sum()) / pore_mask.size
    L = im.shape[axis]

    v_flow_lu = v_lu[..., axis]
    u_darcy_lu = float(np.mean(v_flow_lu))
    u_pore_lu = float(np.mean(v_flow_lu[pore_mask]))
    grad_P_lu = (solver._rho_in - solver._rho_out) * _d3q19.cs2 / L
    k_lu = u_darcy_lu * _d3q19.nu / grad_P_lu

    # Convert to physical units
    dx = voxel_size
    k = k_lu * dx**2
    lu_to_phys = dx / solver.dt
    u_darcy = u_darcy_lu * lu_to_phys
    u_pore = u_pore_lu * lu_to_phys

    kinematic_pressure = solver.kinematic_pressure
    pressure = solver.pressure if rho is not None else None

    return PermeabilityResult(
        im=im,
        axis=axis,
        porosity=porosity,
        k=k,
        u_darcy=u_darcy,
        u_pore=u_pore,
        velocity=solver.velocity,
        kinematic_pressure=kinematic_pressure,
        pressure=pressure,
        converged=solver.converged,
        n_iterations=solver.n_iterations,
        _velocity_lu=v_lu,
        _rho_lu=rho_lu,
        _rho_out=solver._rho_out,
        _k_lu=k_lu,
        _u_darcy_lu=u_darcy_lu,
        _u_pore_lu=u_pore_lu,
    )
