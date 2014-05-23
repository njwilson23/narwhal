import itertools
import operator
import copy
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from karta import Point, Line, LONLAT
from narwhal.cast import Cast, CastCollection
from . import util
from . import gsw

ccmeanp = util.ccmeanp
ccmeans = util.ccmeans

###### T-S plots #######

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

def plot_ts(*casts, **kwargs):
    """ Plot a T-S diagram from Casts or CastCollections """
    ax = kwargs.get("ax", plt.gca())
    drawlegend = kwargs.get("drawlegend", True)
    contourint = kwargs.get("contourint", 0.5)
    labels = kwargs.get("labels", ["cast "+str(i+1) for i in range(len(casts))])
    styles = kwargs.get("styles", itertools.cycle(("ok", "sr", "db", "^g")))
    tkey = kwargs.get("temperature", "theta")
    skey = kwargs.get("salinity", "sal")
    plotkw = {}
    for key in kwargs:
        if key not in ("drawlegend", "contourint", "labels", "styles"):
            plotkw[key] = kwargs[key]
    plotkw.setdefault("ms", 6)

    for i, cast in enumerate(casts):
        sty = next(styles)
        if isinstance(cast, CastCollection):
            for subcast in cast:
                ax.plot(subcast[skey], subcast[tkey], sty, **plotkw)
            ax.lines[-1].set_label(labels[i])
        else:
            ax.plot(cast[skey], cast[tkey], sty, label=labels[i], **plotkw)

    if len(casts) > 1 and drawlegend:
        ax.legend(loc="best", frameon=False)

    if contourint is not None:
        add_sigma_contours(contourint, ax)
    ax.set_xlabel("Salinity")
    ax.set_ylabel(u"Potential temperature (\u00b0C)")
    return

def add_sigma_contours(contourint, ax=None):
    ax = ax if ax is not None else plt.gca()
    sl = ax.get_xlim()
    tl = ax.get_ylim()
    SA = np.linspace(sl[0], sl[1])
    CT = np.linspace(tl[0], tl[1])
    SIGMA = np.reshape([gsw.rho(sa, ct, 0)-1000 for ct in CT
                                                for sa in SA],
                    (50, 50))
    cc = ax.contour(SA, CT, SIGMA, np.arange(np.floor(SIGMA.min()),
                                             np.ceil(SIGMA.max()), contourint),
                    colors="0.4")
    prec = max(0, int(-np.floor(np.log10(contourint))))
    plt.clabel(cc, fmt="%.{0}f".format(prec))
    return

def add_mixing_line(origin, ax=None, icetheta=0, **kw):
    """ Draw a mixing line from `origin::(sal0, theta0)` to the
    effective potential temperature of ice at potential temperature
    `icetheta`, as given by *Jenkins, 1999*.
    """
    kw.setdefault("linestyle", "--")
    kw.setdefault("color", "black")
    kw.setdefault("linewidth", 1.5)

    L = 335e3
    cp = 4.18e3
    ci = 2.11e3
    ice_eff_theta = 0.0 - L/cp - ci/cp * (0.0 - icetheta)

    ax = ax if ax is not None else plt.gca()
    xl, yl = ax.get_xlim(), ax.get_ylim()
    ax.plot((origin[0], 0.0), (origin[1], ice_eff_theta), **kw)
    ax.set_ylim(yl)
    return

def add_melt_line(origin, ax=None, icetheta=-10, **kw):
    add_mixing_line(origin, ax, icetheta, **kw)
    return

def add_runoff_line(origin, ax=None, **kw):
    ax = ax if ax is not None else plt.gca()
    xl, yl = ax.get_xlim(), ax.get_ylim()
    kw.setdefault("linestyle", "--")
    kw.setdefault("color", "black")
    kw.setdefault("linewidth", 1.5)
    ax.plot((origin[0], 0.0), (origin[1], 0.0), **kw)
    ax.set_xlim(xl)
    ax.set_ylim(yl)
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

def plot_section_properties(cc, prop="temp", ax=None,
                            cntrrc=None,
                            cntrfrc=None,
                            interp_method="linear",
                            ninterp=30,
                            mask=True,
                            bottomkey="depth",
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

    Additional keyword arguments are passed to *both* controur and contourf.
    """
    if ax is None:
        ax = plt.gca()
    if cntrrc is None:
        cntrrc = DEFAULT_CONTOUR
    if cntrfrc is None:
        cntrfrc = DEFAULT_CONTOURF

    ccline = Line([c.coords for c in cc], crs=LONLAT)
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
                        row[start:idx] = meanval
                        start = -1
                if start != -1:
                    rawdata[i,start:] = row[start-1]
            else:
                if i != 0:
                    row = rawdata[i-1]

    intdata = griddata(np.c_[obsx.flatten(), obspres.flatten()],
                           rawdata.flatten(),
                           np.c_[intx.flatten(), intpres.flatten()],
                           method=interp_method)
    intdata = intdata.reshape(intx.shape)

    if mask:
        depth = [cast.properties[bottomkey] for cast in cc]
        zmask1 = np.interp(intx[:,0], cx, depth)
        zmask = intpres.T > np.tile(zmask1, (intx.shape[1], 1))
        zmask = zmask.T
        intdata[zmask] = np.nan

    if len(kw) != 0:
        cntrrc = copy.copy(cntrrc)
        cntrfrc = copy.copy(cntrfrc)
        cntrrc.update(kw)
        cntrfrc.update(kw)

    cm = ax.contourf(intx, intpres, intdata, **cntrfrc)
    cl = ax.contour(intx, intpres, intdata, **cntrrc)
    ax.clabel(cl, fmt="%.2f")

    # Set plot bounds
    presgen = (np.array(c[c.primarykey]) for c in cc)
    validgen = (~np.isnan(c[prop]) for c in cc)
    ymax = max(p[msk][-1] for p,msk in zip(presgen, validgen))
    for x_ in cx:
        ax.plot((x_, x_), (ymax, 0), "--", color="0.3")
    ax.set_ylim((ymax, 0))
    ax.set_xlim((cx[0], cx[-1]))
    return cm

def plot_section_bathymetry(bathymetry, vertices=None, ax=None, maxdistance=0.01):
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
    if "vertices":
        bx = []
        segdist = [0.0]
        depth = []
        vline = Line(vertices, crs=LONLAT)
        for a,b in zip(vertices[:-1], vertices[1:]):
            # find all bathymetry within a threshold
            seg = Line((a,b), crs=LONLAT)
            bcoords = [v for v in zip(bathymetry.line.vertices, bathymetry.depth)
                       if seg.within_distance(Point(v[0], crs=LONLAT), 0.01)]

            # project each point in bbox onto the segment, and record
            # the distance from the origin as bx
            pta = Point(a, crs=LONLAT)
            for xy, z in bcoords:
                p = seg.nearest_on_boundary(Point(xy, crs=LONLAT))
                bx.append(segdist[-1] + p.distance(pta))
                depth.append(z)

            segdist.append(seg.length() + segdist[-1])
        
        depth = sorted(depth, key=lambda i: operator.getitem(bx, depth.index(i)))
        bx.sort()
        
    else:
        bx = np.array(bathymetry.line.cumlength())
        depth = bathymetry.depth
    
    ymax = bathymetry.depth.max()
    ax.fill_between(bx, depth, ymax*np.ones_like(depth), color="0.0")
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

