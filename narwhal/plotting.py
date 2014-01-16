import itertools
import matplotlib.pyplot as plt
import numpy as np
import gsw
from cast import Cast, CastCollection
from karta import Point, Line, LONLAT

def ccmean(cc):
    if False in (all(cc[0]["pres"] == c["pres"]) for c in cc[1:]):
        raise ValueError("casts must share pressure levels")
    p = cc[0]["pres"]
    data = dict()
    # shared keys are those in all casts, minus pressure and botdepth
    sharedkeys = set(cc[0].data.keys()).intersection(
                    *[set(c.data.keys()) for c in cc[1:]]).difference(
                    set(("pres", "botdepth")))
    for key in sharedkeys:
        valarray = np.vstack([c.data[key] for c in cc])
        data[key] = nanmean(valarray, axis=0)
    return Cast(p, **data)

###### T-S plots #######

def plot_ts(*casts, **kwargs):
    labels = kwargs.pop("labels", ["cast "+str(i+1) for i in range(len(casts))])
    contourint = kwargs.pop("contourint", 0.5)
    styles = kwargs.pop("styles", itertools.cycle(("ok", "sr", "db", "hg")))
    drawlegend = kwargs.pop("drawlegend", True)

    for i, cast in enumerate(casts):
        if isinstance(cast, CastCollection):
            cast = ccmean(cast)
        plt.plot(cast["sal"], cast["theta"], styles.next(), ms=6, label=labels[i], **kwargs)

    if len(casts) > 1 and drawlegend:
        plt.legend(loc="best", frameon=False)

    add_sigma_contours(contourint, plt.gca())
    plt.xlabel("Salinity")
    plt.ylabel(u"Potential temperature (\u00b0C)")
    return

def add_sigma_contours(contourint, ax=None):
    ax = ax if ax is not None else plt.gca()
    sl = ax.get_xlim()
    tl = ax.get_ylim()
    SA = np.linspace(sl[0], sl[1])
    CT = np.linspace(tl[0], tl[1])
    SIGMA = np.reshape([gsw.gsw_rho(sa, ct, 0)-1000 for ct in CT
                                                    for sa in SA],
                    (50, 50))
    cc = ax.contour(SA, CT, SIGMA,
                    np.arange(np.floor(SIGMA.min()), np.ceil(SIGMA.max()), contourint),
                    colors="0.4")
    prec = max(0, int(-np.floor(np.log10(contourint))))
    plt.clabel(cc, fmt="%.{0}f".format(prec))
    return

def add_mixing_line(origin, ax=None, icetheta=0):
    """ Draw a mixing line from `origin::(sal0, theta0)` to the
    effective potential temperature of ice at potential temperature
    `icetheta`, as given by *Jenkins, 1999*.
    """
    L = 335e3
    cp = 4.18e3
    ci = 2.11e3
    ice_eff_theta = 0.0 - L/cp - ci/cp * (0.0 - icetheta)

    ax = ax if ax is not None else plt.gca()
    xl, yl = ax.get_xlim(), ax.get_ylim()
    ax.plot((origin[0], 0.0), (origin[1], ice_eff_theta), "--k", linewidth=1.5)
    ax.set_xlim(xl)
    ax.set_ylim(yl)
    return

def add_freezing_line(ax=None, p=0.0, air_sat_fraction=0.1):
    ax = ax if ax is not None else plt.gca()
    SA = np.linspace(*ax.get_xlim())
    ctfreeze = lambda sa: gsw.gsw_ct_freezing(sa, p, air_sat_fraction)
    ptfreeze = np.array([gsw.gsw_pt_from_ct(sa, ctfreeze(sa)) for sa in SA])
    ax.plot(SA, ptfreeze, "-.", color="k", label="Freezing line ({0} dbar)".format(p))
    return

###### Section plots #######

def plot_section_properties(cc, **kw):
    ax = kw.pop("ax", gca())

    ccline = Line([c.coords for c in cc], crs=LONLAT)
    cx = np.array(ccline.cumlength())
    x = r_[cx[0], 0.5*(cx[1:] + cx[:-1]), cx[-1]]
    y = cc[0]["pres"]
    ax.pcolormesh(x, y, cc.asarray("sigma"),
                  vmin=20, vmax=28)

    ymax = max(c["pres"][~np.isnan(c["sigma"])][-1] for c in cc)
    ax.set_ylim((ymax, 0))
    ax.set_xlim((x[0], x[-1]))
    return

def plot_section_bathymetry(bathymetry, **kw):
    ax = kw.pop("ax", gca())

    # The bathymetry x should be plotted with respect to CTD line
    if "vertices" in kw:
        bx = []
        segdist = [0.0]
        depth = []

        vertices = kw["vertices"]
        vline = Line(vertices, crs=LONLAT)
        for a,b in zip(vertices[:-1], vertices[1:]):
            # find all bathymetry within bbox defined by seg
            seg = Line((a,b), crs=LONLAT)
            x0, x1 = sorted((a[0], b[0]))
            y0, y1 = sorted((a[1], b[1]))
            bcoords = filter(lambda v: (x0<=v[0][0]<=x1) and (y0<=v[0][1]<=y1),
                             zip(bathymetry.line.vertices, bathymetry.depth))

            # project each point in bbox onto the segment, and record
            # the distance from the origin as bx
            pta = Point(a, crs=LONLAT)
            for xy, z in bcoords:
                p = seg.nearest_on_boundary(Point(xy, crs=LONLAT))
                bx.append(segdist[-1] + p.distance(pta))
                depth.append(z)

            segdist.append(seg.length() + segdist[-1])

    else:
        bx = np.array(bathymetry.line.cumlength())
        depth = bathymetry.depth

    ymax = bathymetry.depth.max()
    ax.fill_between(bx, depth, ymax*np.ones_like(depth),
                    color="0.3")
    return

def plot_section(cc, bathymetry):
    vertices = [c.coords for c in cc]
    plot_section_properties(cc)
    plot_section_bathymetry(bathymetry, vertices=vertices)
    return
