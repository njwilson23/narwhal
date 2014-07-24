import itertools
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import brewer2mpl
from scipy.interpolate import griddata, CloughTocher2DInterpolator
from scipy import ndimage
from scipy import stats
from karta import Multipoint, Line
import narwhal
from narwhal import CastCollection
from . import plotutil
from .. import gsw

try:
    from karta.crs import crsreg
except ImportError:
    import karta as crsreg
LONLAT_WGS84 = crsreg.LONLAT_WGS84
CARTESIAN = crsreg.CARTESIAN

try:
    import pandas
except ImportError:
    # Fake a dataframe
    class DummyPandas(object):
        DataFrame = type(None)
    pandas = DummyPandas

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
        if isinstance(cast, narwhal.AbstractCastCollection):
            for cast_ in cast:
                num = _plot_profile(num, cast_)
        elif isinstance(cast, narwhal.AbstractCast):
            z = cast[cast.primarykey]
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
        if isinstance(cast, narwhal.AbstractCastCollection):
            for cast_ in cast:
                _plot_coords(ax, cast_, **kw)
        elif isinstance(cast, narwhal.AbstractCast):
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

###### Deprecated APIs ######


###### T-S plots #######

def _ensureiterable(item):
    """ Turn *item* into an infinite lazy iterable. """
    if not hasattr(item, "__iter__") or isinstance(item, str):
        return itertools.repeat(item)
    else:
        return itertools.cycle(item)

def _getiterable(kw, name, default):
    """ Equivalent to dict.get except that it ensures the result is an iterable. """
    return _ensureiterable(kw.get(name, default))

def _castlabeliter():
    i = 0
    while True:
        i += 1
        yield "Cast " + str(i)

def plot_ts(castlikes, ax=None,
            xkey="sal", xlabel="Salinity",
            ykey="theta", ylabel=u"Potential temperature (\u00b0C)",
            drawlegend=True, contourint=None, **kwargs):
    """ Plot a T-S diagram from Casts or CastCollections

    Takes a Cast/CastCollection or an iterable of Cast/CastCollection instances
    as an argument.

    Keyword arguments:
    ------------------

    xkey        The data key to plot along x-axis [default: "sal"]

    ykey        The data key to plot along y-axis [default: "theta"]

    ax          The Axes instance to draw to

    drawlegend  Whether to add a legend

    contourint  Interval for sigma contours (deprecated) [default: None]

    labels      An iterable of strings for the legend

    styles      A single or iterable of matplotlib linestyle strings

    colors      A single or iterable of line/marker colors

    markersizes A single of iterable of marker sizes

    Additional keyword arguments are passed to `plot`
    """
    if ax is None:
        ax = plt.gca()

    if not hasattr(castlikes, "__iter__") or isinstance(castlikes, pandas.DataFrame):
        castlikes = (castlikes,)

    n = min(8, max(3, len(castlikes)))
    defaultcolors = brewer2mpl.get_map("Dark2", "Qualitative", n).hex_colors
    color = _getiterable(kwargs, "color", defaultcolors)
    style = _getiterable(kwargs, "style", ["ok", "sr", "db", "^g"])
    label = _getiterable(kwargs, "label", _castlabeliter())
    markersize = _getiterable(kwargs, "ms", 6)

    plotkws = {"ms": itertools.repeat(6)}
    for key in kwargs:
        if key not in ("label", "style", "color"):
            plotkws[key] = _ensureiterable(kwargs[key])

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

        if isinstance(cast, narwhal.AbstractCastCollection):
            x = np.hstack([np.hstack([subcast[xkey], np.nan]) for subcast in cast])
            y = np.hstack([np.hstack([subcast[ykey], np.nan]) for subcast in cast])
            ax.plot(x, y, sty, **plotkw)
        else:
            ax.plot(cast[xkey], cast[ykey], sty, **plotkw)

    if len(castlikes) > 1 and drawlegend:
        ax.legend(loc="best", frameon=False)

    if contourint is not None:
        add_sigma_contours(contourint, ax)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return ax

def add_sigma_contours(contourint, ax=None, pres=0.0):
    """ Add density contours to a T-S plot """
    ax = ax if ax is not None else plt.gca()
    sl = ax.get_xlim()
    tl = ax.get_ylim()
    SA = np.linspace(sl[0], sl[1])
    CT = np.linspace(tl[0], tl[1])
    SIGMA = np.reshape([gsw.rho(sa, ct, pres)-1000 for ct in CT for sa in SA],
                       (50, 50))

    lev0 = np.sign(SIGMA.min()) * ((abs(SIGMA.min()) // contourint) + 1) * contourint
    levels = np.arange(lev0, SIGMA.max(), contourint)
    cc = ax.contour(SA, CT, SIGMA, levels=levels, colors="0.4")
    prec = 0
    while prec < 3 and round(contourint, prec) != contourint:
        prec += 1
    plt.clabel(cc, fmt="%.{0}f".format(prec))
    return

def plot_ts_average(*casts, **kwargs):
    if all(isinstance(c, narwhal.AbstractCast) for c in casts):
        avgcasts = [ccmeanp(casts)]
    else:
        avgcasts = []
        for cast in casts:
            if isinstance(cast, narwhal.AbstractCast):
                avgcasts.append(cast)
            elif isinstance(cast, narwhal.AbstractCastCollection):
                avgcasts.append(ccmeanp(cast))
            else:
                raise TypeError("argument type must be Cast or CastCollection")
    plot_ts(*avgcasts, **kwargs)
    return

def plot_ts_kde(casts, xkey="sal", ykey="theta", ax=None, bw_method=0.2,
                tres=0.2, sres=0.2):
    """ Plot a kernel density estimate T-S diagram """
    if ax is None:
        ax = plt.gca()

    if not hasattr(casts, "__iter__") or isinstance(casts, pandas.DataFrame):
        casts = (casts,)

    temp = np.hstack([c[ykey] for c in casts])
    sal = np.hstack([c[xkey] for c in casts])
    nm = np.hstack([c.nanmask() for c in casts])

    temp = temp[~nm]
    sal = sal[~nm]

    tmin = min(temp) - 0.5
    tmax = max(temp) + 0.5
    smin = min(sal) - 1
    smax = max(sal) + 1

    kernel = stats.gaussian_kde(np.vstack([temp, sal]), bw_method=bw_method)
    T, S = np.mgrid[tmin:tmax:tres, smin:smax:sres]
    xs = np.vstack([T.ravel(), S.ravel()])
    Z = kernel(xs).reshape(T.shape)

    ax.pcolormesh(S, T, Z, cmap="gist_earth_r")
    return

def add_mixing_line(ptA, ptB, ax=None, **kw):
    """ Draw a mixing line between two points in T-S space, provided as
    tuples::(sal, theta).
    """
    ax = ax if ax is not None else plt.gca()
    kw.setdefault("linestyle", "--")
    kw.setdefault("color", "black")
    kw.setdefault("linewidth", 1.5)

    xl, yl = ax.get_xlim(), ax.get_ylim()
    ax.plot((ptA[0], ptB[0]), (ptA[1], ptB[1]), **kw)
    ax.set_xlim(xl)
    ax.set_ylim(yl)
    return

def add_meltwater_line(origin, ax=None, icetheta=-10, **kw):
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
    add_mixing_line(origin, (0.0, ice_eff_theta), ax=ax, **kw)
    return

def add_runoff_line(origin, ax=None, **kw):
    """ Draw mixing line from `origin::(sal0, theta0)` to glacier runoff,
    assumed to be fresh and at the pressure-melting point.
    """
    add_mixing_line(origin, (0.0, 0.0), **kw)
    return

def add_freezing_line(ax=None, p=0.0, air_sat_fraction=0.1, **kw):
    ax = ax if ax is not None else plt.gca()
    kw.setdefault("linestyle", "--")
    kw.setdefault("color", "k")
    SA = np.linspace(*ax.get_xlim())
    ctfreeze = lambda sa: gsw.ct_freezing(sa, p, air_sat_fraction)
    ptfreeze = np.array([gsw.pt_from_ct(sa, ctfreeze(sa)) for sa in SA])
    ax.plot(SA, ptfreeze, label="Freezing line ({0} dbar)".format(p), **kw)
    return

###### Section plots #######

DEFAULT_CONTOUR = {"colors":    "black"}

DEFAULT_CONTOURF = {"cmap":     plt.cm.gist_ncar,
                    "extend":   "both"}

def _interpolate_section_grid(cc, prop, bottomkey, ninterp, interp_method):
    cx = cc.projdist()
    y = cc[0][cc[0].primarykey]
    Xo, Yo = np.meshgrid(cx, y)
    Yi, Xi = np.meshgrid(y, np.linspace(cx[0], cx[-1], ninterp))
    Zo = cc.asarray(prop)

    # interpolate over NaNs in a way that assumes horizontal correlation
    for (i, row) in enumerate(Zo):
        if np.any(np.isnan(row)):
            if np.any(~np.isnan(row)):
                # find groups of NaNs
                start = -1
                for (idx, val) in enumerate(row):
                    if start == -1 and np.isnan(val):
                        start = idx
                    elif start != -1 and not np.isnan(val):
                        if start == 0:
                            meanval = val
                        else:
                            meanval = 0.5 * (val + row[start-1])
                        Zo[i,start:idx] = meanval
                        start = -1
                if start != -1:
                    Zo[i,start:] = row[start-1]
            else:
                if i != 0:
                    Zo[i] = Zo[i-1]  # Extrapolate down
                else:
                    Zo[i] = Zo[i+1]  # Extrapolate to surface

    Zi = griddata(np.c_[Xo.flatten(), Yo.flatten()],
                           Zo.flatten(),
                           np.c_[Xi.flatten(), Yi.flatten()],
                           method=interp_method)
    Zi = Zi.reshape(Xi.shape)
    return Xi, Yi, Zi

def _interpolate_section_tri(cc, prop, bottomkey):
    cx = cc.projdist()

    # interpolate over NaNs in a way that assumes horizontal correlation
    rawdata = cc.asarray(prop)
    for (i, row) in enumerate(rawdata):
        if np.any(np.isnan(row)):
            if np.any(~np.isnan(row)):
                # find groups of NaNs
                start = -1
                for (idx, val) in enumerate(row):
                    if start == -1 and np.isnan(val):
                        start = idx
                    elif start != -1 and not np.isnan(val):
                        if start == 0:
                            meanval = val
                        else:
                            meanval = 0.5 * (val + row[start-1])
                        rawdata[i,start:idx] = meanval
                        start = -1
                if start != -1:
                    rawdata[i,start:] = row[start-1]
            else:
                if i != 0:
                    rawdata[i] = rawdata[i-1]  # Extrapolate down
                else:
                    rawdata[i] = rawdata[i+1]  # Extrapolate to surface

    X, Y, Z = [], [], []
    for (i, cast) in enumerate(cc):
        d = cast.properties[bottomkey]
        pkey = cast.primarykey
        y = cast[pkey][cast[pkey] < d]      # z where z is less than maximum depth
        Y.extend(y)
        X.extend([cx[i] for _ in range(len(y))])
        Z.extend(rawdata[:,i][cast[pkey] < d])

    tri = mtri.Triangulation(X, Y)

    if np.any(np.isnan(rawdata)):
        print("NaNs remaining")
        rawdata[np.isnan(rawdata)] = 999.0

    return tri, Z

def _interpolate_section_cloughtocher(cc, prop, bottomkey, ninterp):
    yo = np.vstack([cast[cast.primarykey] for cast in cc]).T
    xo = np.tile(cc.projdist(), (len(yo), 1))
    zo = cc.asarray(prop)
    msk = ~np.isnan(xo + yo + zo)

    ct2i = CloughTocher2DInterpolator(np.c_[xo[msk], yo[msk]], zo[msk])

    Xi, Yi = np.meshgrid(np.linspace(xo[0,0], xo[0,-1], ninterp),
                         cc[0][cc[0].primarykey])
    Zi = ct2i(Xi, Yi)

    # interpolate over NaNs in a way that assumes horizontal correlation
    for (i, row) in enumerate(Zi):
        if np.any(np.isnan(row)):
            if np.any(~np.isnan(row)):
                # find groups of NaNs
                start = -1
                for (idx, val) in enumerate(row):
                    if start == -1 and np.isnan(val):
                        start = idx
                    elif start != -1 and not np.isnan(val):
                        if start == 0:
                            meanval = val
                        else:
                            meanval = 0.5 * (val + row[start-1])
                        Zi[i,start:idx] = meanval
                        start = -1
                if start != -1:
                    Zi[i,start:] = row[start-1]
            else:
                if i != 0:
                    Zi[i] = Zi[i-1]  # Extrapolate down
                else:
                    Zi[i] = Zi[i+1]  # Extrapolate to surface
    return Xi, Yi, Zi

def _set_section_bounds(ax, cc, prop):
    zgen = (np.array(c[c.primarykey]) for c in cc)
    validmsk = (~np.isnan(c[prop]) for c in cc)
    ymax = max(p[msk][-1] for p,msk in zip(zgen, validmsk))
    cx = cc.projdist()
    for x_ in cx:
        ax.plot((x_, x_), (ymax, 0), "--", color="0.3")
    ax.set_ylim((ymax, 0))
    ax.set_xlim((cx[0], cx[-1]))
    return

def _handle_contour_options(cntrrc, cntrfrc, kw):
    if cntrrc is None:
        cntrrc = DEFAULT_CONTOUR
    if cntrfrc is None:
        cntrfrc = DEFAULT_CONTOURF

    if len(kw) != 0:
        cntrrc = copy.copy(cntrrc)
        cntrfrc = copy.copy(cntrfrc)
        cntrrc.update(kw)
        cntrfrc.update(kw)
    return cntrrc, cntrfrc

def plot_section_properties(cc, prop="temp", ax=None,
                            cntrrc=None,
                            cntrfrc=None,
                            interp_method="linear",
                            ninterp=30,
                            mask=True,
                            bottomkey="depth",
                            kernelsize=None,
                            clabelfmt="%.2f",
                            clabelmanual=False,
                            **kw):
    """ Add water properties from a CastCollection to a section plot.

    Keyword arguments:
    ------------------
    ax                  specific Axes instance to plot on
    prop                Cast property to show
    cntrrc              dictionary of pyplot.contour keyword arguments
    cntrfrc             dictionary of pyplot.contourf keyword argument
    interp_method       method used by scipy.griddata
    mask                apply a NaN mask to the bottom of the section plot
    bottomkey           key in properties giving bottom depth
    kernelsize          smooth the property field using a moving average
                        gaussian to attenuate artefacts.
                        make this as small as possible that still gives
                        reasonable results [default: None]
    clabelfmt           format string for clabel [default: "%.2f"]
    clabelmanual        pass `manual=True` to clabel [default: False]

    Additional keyword arguments are passed to *both* controur and contourf.
    """
    if ax is None:
        ax = plt.gca()
    cntrrc, cntrfrc = _handle_contour_options(cntrrc, cntrfrc, kw)
    (Xi, Yi, Zi) = _interpolate_section_cloughtocher(cc, prop,
                                        bottomkey, ninterp)
    #(Xi, Yi, Zi) = _interpolate_section_grid(cc, prop,
    #                                    bottomkey, ninterp, interp_method)

    if kernelsize is not None:
        Zi = ndimage.filters.gaussian_filter(Zi, kernelsize)

    if mask:
        depth = [cast.properties[bottomkey] for cast in cc]
        cx = cc.projdist()
        base = np.interp(Xi[0,:], cx, depth)
        zmask = Yi > np.tile(base, (Xi.shape[0], 1))
        Zi[zmask] = np.nan

    #cm = ax.contourf(Xi, Yi, Zi, **cntrfrc)
    cm = ax.pcolormesh(Xi, Yi, Zi, vmin=np.nanmin(Zi), vmax=np.nanmax(Zi))
    cl = ax.contour(Xi, Yi, Zi, **cntrrc)

    _set_section_bounds(ax, cc, prop)
    # ax.clabel(cl, fmt=clabelfmt, manual=clabelmanual)
    return cm, cl

def plot_section_properties_tri(cc, prop="temp", ax=None,
                                cntrrc=None,
                                cntrfrc=None,
                                mask=True,
                                bottomkey="depth",
                                clabelfmt="%.2f",
                                clabelmanual=False,
                                **kw):
    """ Add water properties from a CastCollection to a section plot and show
    using tricontourf. Does not indicate no data, so consider using
    `plot_section_properties` instead.

    Keyword arguments:
    ------------------
    ax                  specific Axes instance to plot on
    prop                Cast property to show
    cntrrc              dictionary of pyplot.contour keyword arguments
    cntrfrc             dictionary of pyplot.contourf keyword argument
    interp_method       method used by scipy.griddata
    mask                apply a NaN mask to the bottom of the section plot
    bottomkey           key in properties giving bottom depth

    Additional keyword arguments are passed to *both* controur and contourf.
    """
    if ax is None:
        ax = plt.gca()
    cntrrc, cntrfrc = _handle_contour_options(cntrrc, cntrfrc, kw)
    (tri, Z) = _interpolate_section_tri(cc, prop, bottomkey)

    cm = ax.tricontourf(tri, Z, **cntrfrc)
    cl = ax.tricontour(tri, Z, **cntrrc)

    _set_section_bounds(ax, cc, prop)
    # ax.clabel(cl, fmt=clabelfmt, manual=clabelmanual)
    return cm, cl

def plot_section_bathymetry(bathymetry, vertices, ax=None,
                            maxdistance=200.0, 
                            maxdepth=None,
                            **kw):
    """ Add bathymetry from a Bathymetry object to a section plot.

    Arguments:
    ----------
    bathymetry      a Bathymetry2d instance
    vertices        a CastCollection, karta.Line, or list of points defining a path

    Keyword arguments:
    ------------------
    ax              specific Axes to use
    maxdistance     the maximum distance a bathymetric observation
                    may be from a point in `vertices` to be plotted

    Additional keyword arguments are sent to `plt.fill_between`
    """
    if ax is None:
        ax = plt.gca()

    if isinstance(vertices, Multipoint):
        vertices = vertices.get_vertices()
    elif isinstance(vertices, CastCollection):
        vertices = vertices.coords.get_vertices()

    cruiseline = Line(vertices, crs=LONLAT_WGS84)
    xalong, xacross = bathymetry.project_along_cruise(cruiseline)
    depth_ = bathymetry.depth
    mask = ~np.isnan(depth_) * (xacross < maxdistance)
    depth = depth_[mask]
    xalong = xalong[mask]
    xacross = xacross[mask]

    if maxdepth is None:
        maxdepth = np.nanmax(depth)
    kw.setdefault("color", "0.0")
    ax.fill_between(xalong, depth, maxdepth*np.ones_like(depth), zorder=8, **kw)
    return

def plot_section(cc, bathymetry, ax=None, **kw):
    """ Convenience function to construct a hydrographic section plot by
    calling `plot_section_properties` followed by `plot_section_bathymetry`.
    See those functions for keyword arguments.
    """
    vertices = [c.coords for c in cc]
    plot_section_properties(cc, ax=ax, **kw)
    plot_section_bathymetry(bathymetry, vertices=vertices, ax=ax)
    return

