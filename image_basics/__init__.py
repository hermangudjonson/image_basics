"""basic image routines"""

from loguru import logger

from image_basics import utils

# disable logging for library use
logger.disable("image_basics")

# set cache
utils.set_caches()
