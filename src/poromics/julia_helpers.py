import shutil
from pathlib import Path

import juliapkg
from juliapkg.deps import can_skip_resolve
from juliapkg.find_julia import find_julia
from loguru import _defaults, logger
from tqdm.auto import tqdm

from .utils import suppress_output

# Remove milliseconds from loguru format for cleaner output
logger.remove()
logger_fmt = _defaults.LOGURU_FORMAT
logger_fmt = logger_fmt.replace(".SSS", "")
logger.add(lambda msg: tqdm.write(msg, end=""), format=logger_fmt, colorize=True)


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


def init_julia(quiet: bool = False) -> "juliacall.ModuleValue":
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


def import_package(
    package_name: str, Main: "juliacall.ModuleValue", error: bool = False
) -> "juliacall.ModuleValue":
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


def import_backend(Main: "juliacall.ModuleValue" = None) -> "juliacall.ModuleValue":
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


def is_backend_installed(Main: "juliacall.ModuleValue" = None, error: bool = False) -> bool:
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
