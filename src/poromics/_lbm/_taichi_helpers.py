# Lazy Taichi initialization with GPU auto-detection and CPU fallback.
from loguru import logger

_ti = None


def ensure_taichi():
    """Lazily import and initialize Taichi on first call.

    Attempts GPU backends first (CUDA, Metal, Vulkan), falls back to
    CPU if no GPU is available. Returns the ``taichi`` module.
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
        logger.info("Taichi initialized with GPU backend.")
    except Exception:
        ti.init(arch=ti.cpu, default_fp=ti.f32)
        logger.warning("No GPU backend available, falling back to CPU.")
    _ti = ti
    return _ti
