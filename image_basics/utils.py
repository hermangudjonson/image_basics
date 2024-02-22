"""General package utilities
"""

import os
from pathlib import Path

# define input and working directories depending on platform
ON_KAGGLE: bool = os.environ["PWD"] == "/kaggle/working"
CACHE_DIR = (
    Path("/kaggle/temp")
    if ON_KAGGLE
    else Path("/Users/hermangudjonson/Dropbox/ml_practice/pytorch/image_practice/data")
)
HF_CACHE_DIR = CACHE_DIR / "hf"
WORKING_DIR = (
    Path("/kaggle/working")
    if ON_KAGGLE
    else Path("/Users/hermangudjonson/Dropbox/ml_practice/pytorch/image_practice")
)


def set_caches():
    os.environ["HF_HOME"] = str(HF_CACHE_DIR)
