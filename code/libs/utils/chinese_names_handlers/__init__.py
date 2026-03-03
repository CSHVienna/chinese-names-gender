# Expose the modules themselves (they define functions, not a same-named object)
from . import characternames_handler
from . import pinyinnames_handler

__all__ = [
    "characternames_handler",
    "pinyinnames_handler",
]
