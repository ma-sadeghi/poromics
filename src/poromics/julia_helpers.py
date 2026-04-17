from __future__ import annotations

import os
import platform
import shutil
from pathlib import Path
from typing import Any

import juliapkg
from juliapkg.deps import can_skip_resolve
from juliapkg.find_julia import find_julia
from loguru import logger

from .utils import suppress_output

# Platform → Tortuosity.jl GPU backend package. Users on unusual combinations
# (e.g. Linux + AMD GPU) can set POROMICS_GPU_BACKEND=amdgpu to override.
_GPU_BACKENDS_BY_PLATFORM = {
    ("Darwin", "arm64"): "Metal",
    ("Linux", "x86_64"): "CUDA",
    ("Windows", "AMD64"): "CUDA",
}
_GPU_BACKEND_ALIASES = {"metal": "Metal", "cuda": "CUDA", "amdgpu": "AMDGPU"}


def install_julia(quiet: bool = False) -> None:
    """Installs Julia using juliapkg.

    Args:
        quiet: If True, suppresses output during installation. Default is False.
    """
    # Importing juliacall automatically installs Julia using juliapkg
    if quiet:
        with suppress_output():
            import juliacall  # noqa: F401
    else:
        import juliacall  # noqa: F401


def install_backend(quiet: bool = False) -> None:
    """Installs Julia dependencies for Poromics.

    Args:
        quiet: If True, suppresses output during installation. Default is False.

    Raises:
        ImportError: If Julia is not installed.
    """
    is_julia_installed(error=True)

    if quiet:
        with suppress_output():
            juliapkg.resolve()
    else:
        juliapkg.resolve()


def init_julia(quiet: bool = False) -> Any:
    """Initializes Julia and returns the Main module.

    Args:
        quiet: If True, suppresses the output of Julia initialization. Default is False.

    Returns:
        Main: The Julia Main module.

    Raises:
        ImportError: If Julia is not installed.
    """
    is_julia_installed(error=True)
    if not can_skip_resolve():
        logger.warning("Julia is installed, but needs to be resolved...")
    if quiet:
        with suppress_output():
            from juliacall import Main  # type: ignore
    else:
        from juliacall import Main  # type: ignore

    return Main


def import_package(package_name: str, Main: Any, error: bool = False) -> Any:
    """Imports a package in Julia and returns the module.

    Args:
        package_name: Name of the Julia package to import.
        Main: Julia Main module.
        error: If True, raises an error if the package is not found. Default is False.

    Returns:
        package: Handle to the imported package.

    Raises:
        ImportError: If the package is not found and error is True.
    """
    from juliacall import JuliaError

    try:
        Main.seval(f"using {package_name}")
        return eval(f"Main.{package_name}")
    except JuliaError as e:
        if error:
            raise e
    return None


def import_backend(Main: Any = None) -> Any:
    """Imports Tortuosity.jl package from Julia.

    Args:
        Main: Julia Main module. Default is None. If None, the Main module will
            be initialized.

    Returns:
        backend: Handle to the Tortuosity.jl package.

    Raises:
        ImportError: If Julia is not installed or the package is not found.
    """
    Main = init_julia() if Main is None else Main
    is_backend_installed(Main=Main, error=True)
    return import_package("Tortuosity", Main)


def is_julia_installed(error: bool = False) -> bool:
    """Checks that Julia is installed.

    Args:
        error: If True, raises an error if Julia is not found. Default is False.

    Returns:
        flag: True if Julia is installed, False otherwise.

    Raises:
        ImportError: If Julia is not installed and error is True.
    """
    # Look for system-wide Julia executable
    try:
        find_julia()
        return True
    except Exception:
        pass
    # Look for local Julia executable (e.g., installed by juliapkg)
    if can_skip_resolve():
        return True
    msg = "Julia not found. Visit https://github.com/JuliaLang/juliaup and install Julia."
    if error:
        raise ImportError(msg)
    return False


def is_backend_installed(Main: Any = None, error: bool = False) -> bool:
    """Checks if Tortuosity.jl is installed.

    Args:
        Main: Julia Main module. Default is None. If None, it will be initialized.
        error: If True, raises an error if backend is not found. Default is False.

    Returns:
        flag: True if the package is installed, False otherwise.

    Raises:
        ImportError: If Julia is not installed or backend is not found and error is True.
    """
    Main = init_julia() if Main is None else Main
    if import_package("Tortuosity", Main, error=False) is not None:
        return True
    msg = "Tortuosity.jl not found, run 'python -m poromics install'"
    if error:
        raise ImportError(msg)
    return False


def ensure_julia_deps_ready(quiet: bool = False, retry: bool = True) -> None:
    """Ensures Julia and Tortuosity.jl are installed.

    Args:
        quiet: If True, suppresses output during installation. Default is False.
        retry: If True, retries the installation if it fails. Default is True.

    Raises:
        ImportError: If Julia or Tortuosity.jl cannot be installed.
    """

    def _ensure_julia_deps_ready(quiet):
        if not is_julia_installed(error=False):
            logger.warning("Julia not found, installing Julia...")
            install_julia(quiet=quiet)
        Main = init_julia(quiet=quiet)
        if not is_backend_installed(Main=Main, error=False):
            logger.warning("Julia dependencies not found, installing Tortuosity.jl...")
            install_backend(quiet=quiet)

    def _reset_julia_env(quiet):
        remove_julia_env()
        if quiet:
            with suppress_output():
                juliapkg.resolve(force=True)
        else:
            juliapkg.resolve(force=True)

    try:
        _ensure_julia_deps_ready(quiet)
    except Exception:
        if retry:
            _reset_julia_env(quiet)
            _ensure_julia_deps_ready(quiet)
            return
        raise


def _detect_gpu_backend() -> str | None:
    """Return the GPU backend package for this platform, or None if unsupported.

    ``POROMICS_GPU_BACKEND=metal|cuda|amdgpu`` overrides the platform default.
    """
    override = os.environ.get("POROMICS_GPU_BACKEND", "").strip()
    if override:
        return _GPU_BACKEND_ALIASES.get(override.lower(), override)
    return _GPU_BACKENDS_BY_PLATFORM.get((platform.system(), platform.machine()))


def ensure_gpu_backend(Main: Any) -> str | None:
    """Install and load a GPU backend in Julia; verify the device is functional.

    Returns the backend package name (e.g. ``"Metal"``) when a usable GPU is
    available, or ``None`` when the caller should fall back to CPU. All
    fallback paths emit a warning explaining why.

    Parameters
    ----------
    Main : juliacall Main module

    Returns
    -------
    backend : str or None
        ``"Metal"``, ``"CUDA"``, ``"AMDGPU"``, or ``None``.
    """
    from juliacall import JuliaError

    backend = _detect_gpu_backend()
    if backend is None:
        logger.warning(
            f"No GPU backend mapping for {platform.system()}/{platform.machine()}. "
            "Set POROMICS_GPU_BACKEND={metal,cuda,amdgpu} to override. "
            "Falling back to CPU."
        )
        return None
    try:
        Main.seval(f"using {backend}")
    except JuliaError:
        try:
            Main.seval(f'import Pkg; Pkg.add("{backend}"); using {backend}')
        except JuliaError as e:
            logger.warning(f"Failed to install/load {backend}.jl: {e}. Falling back to CPU.")
            return None
    # Upstream Tortuosity{Backend}Ext.__init__ registers the backend only when
    # `.functional()` is true. When the package loads but the hardware is not
    # usable, Tortuosity errors later with a "no GPU backend is registered"
    # message that doesn't mention the real cause. See
    # https://github.com/ma-sadeghi/Tortuosity.jl/issues/79 — we pre-check here
    # so the user gets a meaningful warning before falling back to CPU.
    try:
        functional = bool(Main.seval(f"{backend}.functional()"))
    except JuliaError:
        functional = False
    if not functional:
        logger.warning(
            f"{backend}.jl loaded but {backend}.functional() returned false "
            "(no working GPU device detected). Falling back to CPU."
        )
        return None
    return backend


def remove_julia_env() -> None:
    """Removes the active Julia environment directory.

    When Julia or its dependencies are corrupted, this is a possible fix.
    """
    path_julia_env = Path(juliapkg.project())

    if path_julia_env.exists():
        logger.warning(f"Removing Julia environment directory: {path_julia_env}")
        shutil.rmtree(path_julia_env)
    else:
        logger.warning("Julia environment directory not found.")
