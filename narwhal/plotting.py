import itertools
import operator
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import brewer2mpl
from scipy.interpolate import griddata
from scipy import ndimage
from scipy import stats
from karta import Point, Line, LONLAT_WGS84, CARTESIAN
from narwhal.cast import CastCollection
from . import plotutil
from . import gsw

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

ccmeanp = plotutil.ccmeanp
ccmeans = plotutil.ccmeans

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

def plot_ts(castlikes, xkey="sal", ykey="theta", ax=None,
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
    # xkey = kwargs.pop("xkey", "sal")
    # ykey = kwargs.pop("ykey", "theta")
    # ax = kwargs.pop("ax", plt.gca())
    # drawlegend = kwargs.pop("drawlegend", True)
    # contourint = kwargs.pop("contourint", None)
    if ax is None:
        ax = plt.gca()

    if not hasattr(castlikes, "__iter__"):
        castlikes = (castlikes,)
    
    label = _getiterable(kwargs, "label",
                         ["Cast "+str(i+1) for i in range(len(castlikes))])
    style = _getiterable(kwargs, "style", ["ok", "sr", "db", "^g"])

    n = min(8, max(3, len(castlikes)))
    defaultcolors = brewer2mpl.get_map("Dark2", "Qualitative", n).hex_colors
    color = _getiterable(kwargs, "color", defaultcolors)
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

        if hasattr(cast, "_type") and cast._type == "castcollection":
            x = np.hstack([np.hstack([subcast[xkey], np.nan]) for subcast in cast])
            y = np.hstack([np.hstack([subcast[ykey], np.nan]) for subcast in cast])
            ax.plot(x, y, sty, **plotkw)
        else:
            ax.plot(cast[xkey], cast[ykey], sty, **plotkw)

    if len(castlikes) > 1 and drawlegend:
        ax.legend(loc="best", frameon=False)

    if contourint is not None:
        add_sigma_contours(contourint, ax)

    ax.set_xlabel("Salinity")
    ax.set_ylabel(u"Potential temperature (\u00b0C)")
    return ax

def add_sigma_contours(contourint, ax=None):
    """ Add density contours to a T-S plot """
    ax = ax if ax is not None else plt.gca()
    sl = ax.get_xlim()
    tl = ax.get_ylim()
    SA = np.linspace(sl[0], sl[1])
    CT = np.linspace(tl[0], tl[1])
    SIGMA = np.reshape([gsw.rho(sa, ct, 0)-1000 for ct in CT for sa in SA],
                       (50, 50))

    lev0 = np.sign(SIGMA.min()) * ((abs(SIGMA.min()) // contourint) + 1) * contourint
    levels = np.arange(lev0, SIGMA.max(), contourint)
    cc = ax.contour(SA, CT, SIGMA, levels=levels, colors="0.4")
    prec = max(0, int(-np.floor(np.log10(contourint))))
    plt.clabel(cc, fmt="%.{0}f".format(prec))
    return

def plot_ts_average(*casts, **kwargs):
    if False not in map(lambda c: c._type == "cast", casts):
        avgcasts = [ccmeanp(casts)]
    else:
        avgcasts = []
        for cast in casts:
            if cast._type == "cast":
                avgcasts.append(cast)
            elif cast._type == "ctd_collection":
                avgcasts.append(ccmeanp(cast))
            else:
                raise TypeError("argument is neither a cast nor a castcollection")
    plot_ts(*avgcasts, **kwargs)
    return

def plot_ts_kde(casts, xkey="sal", ykey="theta", ax=None, bw_method=0.2,
                tres=0.2, sres=0.2):
    """ Plot a kernel density estimate T-S diagram """
    if ax is None:
        ax = plt.gca()
    
    if not hasattr(casts, "__iter__"):
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
    ccline = Line([c.coords for c in cc], crs=LONLAT_WGS84)
    cx = np.array(ccline.cumlength())
    y = cc[0][cc[0].primarykey]
    obsx, obspres = np.meshgrid(cx, y)
    intpres, intx = np.meshgrid(y, np.linspace(cx[0], cx[-1], ninterp))
    rawdata = cc.asarray(prop)

    # interpolate over NaNs in a way that assumes horizontal correlation
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

    intdata = griddata(np.c_[obsx.flatten(), obspres.flatten()],
                           rawdata.flatten(),
                           np.c_[intx.flatten(), intpres.flatten()],
                           method=interp_method)
    intdata = intdata.reshape(intx.shape)
    return intx, intpres, intdata, cx

def _interpolate_section_tri(cc, prop, bottomkey):
    ccline = Line([c.coords for c in cc], crs=LONLAT_WGS84)
    cx = np.array(ccline.cumlength())

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

    return tri, Z, cx

def _set_section_bounds(ax, cc, cx, prop):
    zgen = (np.array(c[c.primarykey]) for c in cc)
    validmsk = (~np.isnan(c[prop]) for c in cc)
    ymax = max(p[msk][-1] for p,msk in zip(zgen, validmsk))
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
    (intx, intpres, intdata, cx) = _interpolate_section_grid(cc, prop,
                                        bottomkey, ninterp, interp_method)

    if kernelsize is not None:
        intdata = ndimage.filters.gaussian_filter(intdata, kernelsize)

    if mask:
        depth = [cast.properties[bottomkey] for cast in cc]
        zmask1 = np.interp(intx[:,0], cx, depth)
        zmask = intpres.T > np.tile(zmask1, (intx.shape[1], 1))
        zmask = zmask.T
        intdata[zmask] = np.nan

    cm = ax.contourf(intx, intpres, intdata, **cntrfrc)
    cl = ax.contour(intx, intpres, intdata, **cntrrc)

    _set_section_bounds(ax, cc, cx, prop)
    ax.clabel(cl, fmt=clabelfmt, manual=clabelmanual)
    return cm

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
    (tri, Z, cx) = _interpolate_section_tri(cc, prop, bottomkey)

    cm = ax.tricontourf(tri, Z, **cntrfrc)
    cl = ax.tricontour(tri, Z, **cntrrc)

    _set_section_bounds(ax, cc, cx, prop)
    ax.clabel(cl, fmt=clabelfmt, manual=clabelmanual)
    return cm

def plot_section_bathymetry(bathymetry, vertices=None, ax=None, maxdistance=200.0,
                            crs=LONLAT_WGS84):
    """ Add bathymetry from a Bathymetry object to a section plot.
    
    Keyword arguments:
    ------------------
    ax                  specific Axes to use
    vertices            a list of points defining a cruise path
    maxdistance         the maximum distance a bathymetric observation
                        may be from a point in `vertices` to be plotted
    """
    if ax is None:
        ax = plt.gca()
    
    # The bathymetry x should be plotted with respect to CTD line
    if vertices:
        bx = []
        segdist = [0.0]
        depth = []
        bathpts = [pt for pt in bathymetry]
        for a,b in zip(vertices[:-1], vertices[1:]):
            # find all bathymetry within a threshold
            seg = Line((a,b), crs=crs)
            bcoords = [v for v in zip(bathpts, bathymetry.depth)
                         if seg.within_distance(v[0], maxdistance)]

            # project each point in bbox onto the segment, and record
            # the distance from the origin as bx
            pta = Point(a, crs=crs)
            for pt, z in bcoords:
                p = seg.nearest_on_boundary(pt)
                bx.append(segdist[-1] + p.distance(pta))
                depth.append(z)

            segdist.append(seg.length() + segdist[-1])
        
        depth = sorted(depth, key=lambda i: operator.getitem(bx, depth.index(i)))
        bx.sort()
        
    else:
        bx = np.array(bathymetry.cumlength())
        depth = bathymetry.depth
    
    ymax = bathymetry.depth.max()
    ax.fill_between(bx, depth, ymax*np.ones_like(depth), color="0.0", zorder=8)
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

def plot_profiles(castlikes, key="temp", ax=None, **kw):
    """ Plot vertical profiles from casts """
    def _plot_profile(ax, cast):
        if hasattr(cast, "_type"):
            if cast._type == "castcollection":
                for cast_ in cast:
                    _plot_profile(ax, cast_)
            else:
                z = cast[cast.primarykey]
                ax.plot(cast[key], z, **kw)
        else:
            raise TypeError("Argument not Cast or CastCollection-like")
        return

    if ax is None:
        ax = plt.gca()
    if not hasattr(castlikes, "__iter__"):
        castlikes = (castlikes,)
    for castlike in castlikes:
        _plot_profile(ax, castlike)
    if not ax.yaxis_inverted():
        ax.invert_yaxis()
    return ax

def plot_map(castlikes, ax=None, **kw):
    """ Plot a simple map of cast locations. """
    def _plot_coords(ax, cast):
        if hasattr(cast, "_type"):
            if cast._type == "castcollection":
                for cast_ in cast:
                    _plot_coords(ax, cast_)
            else:
                ax.plot(cast.coords[0], cast.coords[1], "ok")
        else:
            raise TypeError("Argument not Cast or CastCollection-like")
        return

    if ax is None:
        ax = plt.gca()
    if not hasattr(castlikes, "__iter__"):
        castlikes = (castlikes,)
    for castlike in castlikes:
        _plot_coords(ax, castlike)
    return ax
