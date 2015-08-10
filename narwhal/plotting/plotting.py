import matplotlib.pyplot as plt

from .colors import default_colors
from . import plotutil
from ..cast import AbstractCast, AbstractCastCollection

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def _castlabeliter():
    i = 0
    while True:
        i += 1
        yield "Cast " + str(i)

def plot_profiles(castlikes, key="temp", ykey="depth", ax=None, **kw):
    """ Plot vertical profiles from casts. Keyword arguments are passed to
    `pyplot.plot`. If keyword arguments are non-string iterables, profiles are
    plotted with the items in order. """

    # guess the number of casts - in the future, get this properly
    n = min(8, max(3, len(castlikes)))

    plotkw = dict((k, plotutil.ensureiterable(v)) for k,v in kw.items())
    plotkw.setdefault("color", plotutil.ensureiterable(default_colors(n)))
    plotkw.setdefault("label", _castlabeliter())

    def _plot_profile(num, cast):
        if isinstance(cast, AbstractCastCollection):
            for cast_ in cast:
                num = _plot_profile(num, cast_)
        elif isinstance(cast, AbstractCast):
            _kw = dict((k, next(v)) for k, v in plotkw.items())
            ax.plot(cast[key], cast[ykey], **_kw)
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

def plot_section(castlikes, prop="temp", cmap=plt.cm.Spectral, **kw):
    """ Convenience method for making a basic section plot. """
    ax = plt.axes(projection="section", axisbg="#404040")
    c = ax.contour(castlikes, prop, colors="black", linestyles="-", linewidths=0.5)
    plt.clabel(c)
    ax.contourf(castlikes, prop, cmap=cmap)
    ax.mark_stations(castlikes)
    ax.label_stations(castlikes, [str(i+1) for i in range(len(castlikes))], vert_offset=5)
    return ax

def plot_map(castlikes, ax=None, crs=None, **kw):
    """ Plot a simple map of cast locations. """
    def _coord_transformer(cast, crs):
        if crs:
            return crs.project(cast.coords[0], cast.coords[1])
        else:
            return (cast.coords[0], cast.coords[1])

    def _plot_coords(ax, cast, **kw):
        if isinstance(cast, AbstractCastCollection):
            for cast_ in cast:
                _plot_coords(ax, cast_, **kw)
        elif isinstance(cast, AbstractCast):
            ax.plot(*_coord_transformer(cast, crs), **kw)
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

