from importlib import metadata

__version__ = metadata.version("onekit")

from .core import *

del (
    metadata,
    core,
)
