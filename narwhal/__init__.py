
from .cast import Cast, CastCollection, read
from .bathymetry import Bathymetry
#from . import plotting
from . import util

__all__ = ["cast", "bathymetry", "gsw", "util"]

try:
    from . import gsw
    __all__ += ["gsw"]
except OSError:
    gsw = None

