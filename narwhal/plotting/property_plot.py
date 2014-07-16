import itertools
import brewer2mpl
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from narwhal import AbstractCastCollection
from .. import gsw

try:
    import pandas
except ImportError:
    # Fake a dataframe
    class DummyPandas(object):
        DataFrame = type(None)
    pandas = DummyPandas

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

class PropertyPropertyAxes(plt.Axes):
    name = "propertyplot"

    def __init__(self, *args, **kwargs):
        super(plt.Axes, self).__init__(*args, **kwargs)

    @staticmethod
    def _ensureiterable(item):
        """ Turn *item* into an infinite lazy iterable. """
        if not hasattr(item, "__iter__") or isinstance(item, str):
            return itertools.repeat(item)
        else:
            return itertools.cycle(item)

    def _getiterable(self, kw, name, default):
        """ Equivalent to dict.get except that it ensures the result is an iterable. """
        return self._ensureiterable(kw.get(name, default))

    @staticmethod
    def _castlabeliter():
        i = 0
        while True:
            i += 1
            yield "Cast " + str(i)

    def plot_casts(self, castlikes, xkey, ykey, xlabel=None,  ylabel=None,
                   drawlegend=True, **kwargs):
        """ Plot a T-S diagram from Casts or CastCollections

        Takes a Cast/CastCollection or an iterable of Cast/CastCollection instances
        as an argument.

        Keyword arguments:
        ------------------

        xkey            The data key to plot along x-axis [default: "sal"]
        ykey            The data key to plot along y-axis [default: "theta"]
        drawlegend      Whether to add a legend
        labels          An iterable of strings for the legend
        styles          A single or iterable of matplotlib linestyle strings
        colors          A single or iterable of line/marker colors
        markersizes     A single of iterable of marker sizes

        Additional keyword arguments are passed to `plot`
        """
        if not hasattr(castlikes, "__iter__") or isinstance(castlikes, pandas.DataFrame):
            castlikes = (castlikes,)
        if xlabel is None:
            xlabel = xkey
        if ylabel is None:
            ylabel = ykey

        n = min(8, max(3, len(castlikes)))
        defaultcolors = brewer2mpl.get_map("Dark2", "Qualitative", n).hex_colors
        color = self._getiterable(kwargs, "color", defaultcolors)
        style = self._getiterable(kwargs, "style", ["ok", "sr", "db", "^g"])
        label = self._getiterable(kwargs, "label", self._castlabeliter())
        markersize = self._getiterable(kwargs, "ms", 6)

        plotkws = {"ms": itertools.repeat(6)}
        for key in kwargs:
            if key not in ("label", "style", "color"):
                plotkws[key] = self._ensureiterable(kwargs[key])

        oldlabels = []
        for i, cast in enumerate(castlikes):
            plotkw = {}
            for key in plotkws:
                plotkw[key] = next(plotkws[key])

            sty = next(style)
            lbl = next(label)
            plotkw["color"] = next(color)
            plotkw["ms"] = next(markersize)
            if lbl not in oldlabels:
                plotkw["label"] = lbl
                oldlabels.append(lbl)

            lines = []
            if isinstance(cast, AbstractCastCollection):
                x = np.hstack([np.hstack([subcast[xkey], np.nan]) for subcast in cast])
                y = np.hstack([np.hstack([subcast[ykey], np.nan]) for subcast in cast])
                lines.append(self.plot(x, y, sty, **plotkw))
            else:
                lines.append(self.plot(cast[xkey], cast[ykey], sty, **plotkw))

        if len(castlikes) > 1 and drawlegend:
            self.legend(loc="best", frameon=False)

        self.set_xlabel(xlabel)
        self.set_ylabel(ylabel)
        return lines

    def plot_ts(self, castlikes, p1="sal", p2="theta",
                xlabel="Salinity", ylabel=u"Potential temperature (\u00b0C)",
                **kw):
        return self.plot_casts(castlikes, p1, p2, xlabel=xlabel, ylabel=ylabel, **kw)

    def add_sigma_contours(self, contourint, pres=0.0):
        """ Add density contours to a T-S plot """
        sl = self.get_xlim()
        tl = self.get_ylim()
        SA = np.linspace(sl[0], sl[1])
        CT = np.linspace(tl[0], tl[1])
        SIGMA = np.reshape([gsw.rho(sa, ct, pres)-1000 for ct in CT for sa in SA],
                           (50, 50))

        lev0 = np.sign(SIGMA.min()) * ((abs(SIGMA.min()) // contourint) + 1) * contourint
        levels = np.arange(lev0, SIGMA.max(), contourint)
        cc = self.contour(SA, CT, SIGMA, levels=levels, colors="0.4")
        prec = 0
        while prec < 3 and round(contourint, prec) != contourint:
            prec += 1
        self.clabel(cc, fmt="%.{0}f".format(prec))
        return

    def add_mixing_line(self, ptA, ptB, **kw):
        """ Draw a mixing line between two points in T-S space, provided as
        tuples::(sal, theta).
        """
        kw.setdefault("linestyle", "--")
        kw.setdefault("color", "black")
        kw.setdefault("linewidth", 1.5)

        xl, yl = self.get_xlim(), self.get_ylim()
        line = self.plot((ptA[0], ptB[0]), (ptA[1], ptB[1]), **kw)
        self.set_xlim(xl)
        self.set_ylim(yl)
        return line

    def add_meltwater_line(self, origin, icetheta=-10, **kw):
        """ Draw a line on a TS plot representing the mixing line between a source
        water at `origin::(sal0, theta0)` and meltwater from an ice mass with
        temperature `icetheta`.

        The effective potential temperature of ice at potential temperature
        `icetheta` is used as given by *Jenkins, 1999*.
        """
        L = 335e3
        cp = 4.18e3
        ci = 2.11e3
        ice_eff_theta = 0.0 - L/cp - ci/cp * (0.0 - icetheta)
        self.add_mixing_line(origin, (0.0, ice_eff_theta), **kw)
        return

    def add_runoff_line(self, origin, **kw):
        """ Draw mixing line from `origin::(sal0, theta0)` to glacier runoff,
        assumed to be fresh and at the pressure-melting point.
        """
        self.add_mixing_line(origin, (0.0, 0.0), **kw)
        return

    def add_freezing_line(self, p=0.0, air_sat_fraction=0.1, **kw):
        kw.setdefault("linestyle", "--")
        kw.setdefault("color", "k")
        SA = np.linspace(*self.get_xlim())
        ctfreeze = lambda sa: gsw.ct_freezing(sa, p, air_sat_fraction)
        ptfreeze = np.array([gsw.pt_from_ct(sa, ctfreeze(sa)) for sa in SA])
        self.plot(SA, ptfreeze, label="Freezing line ({0} dbar)".format(p), **kw)
        return

def plot_ts(*args, **kwargs):
    """ Convenience function for making TS plots

    WIP
    """
    ax = plt.axes(projection="propertyplot")
    return ax.plot_ts(*args, **kwargs)

matplotlib.projections.register_projection(PropertyPropertyAxes)
