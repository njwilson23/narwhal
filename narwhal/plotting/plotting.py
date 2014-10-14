import itertools
import copy
import numpy as np
import pandas
from scipy.interpolate import griddata, CloughTocher2DInterpolator
from scipy import ndimage
from scipy import stats
from karta import Multipoint, Line

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import brewer2mpl

from ..cast import AbstractCast, AbstractCastCollection
from . import plotutil
from .. import gsw

try:
    from karta.crs import crsreg
except ImportError:
    import karta as crsreg
LONLAT_WGS84 = crsreg.LONLAT_WGS84
CARTESIAN = crsreg.CARTESIAN

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

ccmeanp = plotutil.ccmeanp
ccmeans = plotutil.ccmeans

def plot_profiles(castlikes, key="temp", ax=None, **kw):
    """ Plot vertical profiles from casts """
    # guess the number of casts - in the future, get this properly
    n = min(8, max(3, len(castlikes)))
    defaultcolors = brewer2mpl.get_map("Dark2", "Qualitative", n).hex_colors

    plotkw = dict((k, _ensureiterable(v)) for k,v in kw.items())
    plotkw.setdefault("color", _ensureiterable(defaultcolors))
    plotkw.setdefault("label", _castlabeliter())

    def _plot_profile(num, cast):
        if isinstance(cast, AbstractCastCollection):
            for cast_ in cast:
                num = _plot_profile(num, cast_)
        elif isinstance(cast, AbstractCast):
            z = cast[cast.zname]
            _kw = dict((k, next(v)) for k, v in plotkw.items())
            ax.plot(cast[key], z, **_kw)
            num += 1
        else:
            raise TypeError("Argument not a Cast or CastCollection")
        return num

    if ax is None:
        ax = plt.gca()
    num = 0
    if not hasattr(castlikes, "__iter__"):
        castlikes = (castlikes,)
    for castlike in castlikes:
        num = _plot_profile(num, castlike)
    if not ax.yaxis_inverted():
        ax.invert_yaxis()
    return ax

def plot_map(castlikes, ax=None, **kw):
    """ Plot a simple map of cast locations. """
    def _plot_coords(ax, cast, **kw):
        if isinstance(cast, AbstractCastCollection):
            for cast_ in cast:
                _plot_coords(ax, cast_, **kw)
        elif isinstance(cast, AbstractCast):
            ax.plot(cast.coords[0], cast.coords[1], **kw)
        else:
            raise TypeError("Argument not Cast or CastCollection-like")
        return

    if ax is None:
        ax = plt.gca()
    if not hasattr(castlikes, "__iter__"):
        castlikes = (castlikes,)
    kw.setdefault("marker", "o")
    kw.setdefault("color", "k")
    for castlike in castlikes:
        _plot_coords(ax, castlike, **kw)
    return ax

