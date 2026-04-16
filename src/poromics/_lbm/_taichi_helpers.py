# Lazy Taichi initialization with GPU auto-detection and CPU fallback.
from loguru import logger

_ti = None


def _describe_arch(ti):
    """Return a human-readable name for the currently active Taichi arch."""
    try:
        arch = ti.lang.impl.current_cfg().arch
        return str(arch).split(".")[-1]
    except Exception:
        return "unknown"


def ensure_taichi():
    """Lazily import and initialize Taichi on first call.

    Attempts GPU backends first (CUDA / Metal / Vulkan / OpenGL), falls
    back to CPU if no GPU is available. When the fallback happens,
    logs a WARNING including the underlying error so users who expect
    GPU don't silently run on CPU. Returns the ``taichi`` module.
    """
    global _ti
    if _ti is not None:
        return _ti
    try:
        import taichi as ti
    except ImportError:
        msg = "taichi is required for LBM solvers. Install it with: pip install taichi"
        raise ImportError(msg)
    try:
        ti.init(arch=ti.gpu, default_fp=ti.f32)
        logger.info(f"Taichi initialized on GPU backend: {_describe_arch(ti)}")
    except Exception as gpu_err:
        ti.init(arch=ti.cpu, default_fp=ti.f32)
        logger.warning(
            f"GPU backend unavailable, falling back to Taichi CPU "
            f"({_describe_arch(ti)}). Original error: "
            f"{type(gpu_err).__name__}: {gpu_err}"
        )
    _ti = ti
    return _ti
