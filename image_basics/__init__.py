"""basic image routines"""

from loguru import logger

# provide submodule api
import image_basics.data
import image_basics.model
import image_basics.plot
import image_basics.tasks
import image_basics.train
import image_basics.utils  # noqa: F401
from image_basics import utils

# disable logging for library use
logger.disable("image_basics")

# set cache
utils.set_caches()
