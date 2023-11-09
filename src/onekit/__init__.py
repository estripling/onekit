from importlib import metadata

__version__ = metadata.version("onekit")

from .pytlz import *

del (
    metadata,
    pytlz,
)
