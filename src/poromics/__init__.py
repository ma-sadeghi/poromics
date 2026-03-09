from loguru import logger

# Configure loguru: WARNING+ only, no milliseconds in timestamps.
logger.remove()
logger.add(
    lambda msg: __import__("sys").stderr.write(msg),
    format="{time:HH:mm:ss} | <level>{level:<8}</level> | {name}:{function}:{line} - {message}",
    colorize=True,
    level="WARNING",
)

from ._metrics import *  # noqa: F403, E402
from . import simulation  # noqa: F401, E402
from .version import __version__  # noqa: F401, E402
