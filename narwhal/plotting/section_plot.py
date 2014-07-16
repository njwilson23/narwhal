import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# import matplotlib.tri as mtri
from scipy.interpolate import griddata, CloughTocher2DInterpolator
from scipy import ndimage
from karta import Line
from narwhal import AbstractCast, AbstractCastCollection

try:
    from karta.crs import crsreg
except ImportError:
    import karta as crsreg
LONLAT_WGS84 = crsreg.LONLAT_WGS84
CARTESIAN = crsreg.CARTESIAN

class BaseSectionAxes(plt.Axes):

    def __init__(self, *args, **kwargs):
        super(plt.Axes, self).__init__(*args, **kwargs)

    def _set_section_bounds(self, cc, prop):
        zgen = (np.array(c[c.primarykey]) for c in cc)
        validmsk = (~np.isnan(c[prop]) for c in cc)
        ymax = max(p[msk][-1] for p,msk in zip(zgen, validmsk))
        cx = cc.projdist()
        for x_ in cx:
            self.plot((x_, x_), (ymax, 0), "--", color="0.3")
        self.set_ylim((ymax, 0))
        self.set_xlim((cx[0], cx[-1]))
        return

    @staticmethod
    def _smooth_field(d, sk):
        msk = np.isnan(d)
        d[msk] = 0.0
        d_ = ndimage.filters.gaussian_filter(d, sk)
        d_[msk] = np.nan
        return d_

    @staticmethod
    def _computemask(cc, Xi, Yi, Zi, bottomkey):
        depth = [cast.properties[bottomkey] for cast in cc]
        cx = cc.projdist()
        base = np.interp(Xi[0,:], cx, depth)
        zmask = Yi > np.tile(base, (Xi.shape[0], 1))
        return zmask

    def contour(self, cc, prop, ninterp=30, sk=None, mask=True,
                bottomkey="depth", interp_scheme="standard", **kwargs):

        if not isinstance(cc, AbstractCastCollection):
            raise TypeError("first argument must be a CastCollection type")

        (Xi, Yi, Zi) = self._interpolate_section(cc, prop, ninterp, interp_scheme)

        if sk is not None:
            Zi = self._smooth_field(Zi, sk)

        if mask:
            msk = self._computemask(cc, Xi, Yi, Zi, bottomkey)
            Zi[msk] = np.nan

        self._set_section_bounds(cc, prop)
        return super(BaseSectionAxes, self).contour(Xi, Yi, Zi, **kwargs)

    def contourf(self, cc, prop, ninterp=30, sk=None, mask=True,
                 bottomkey="depth", interp_scheme="standard", **kwargs):

        if not isinstance(cc, AbstractCastCollection):
            raise TypeError("first argument must be a CastCollection type")

        (Xi, Yi, Zi) = self._interpolate_section(cc, prop, ninterp, interp_scheme)

        if sk is not None:
            Zi = self._smooth_field(Zi, sk)

        if mask:
            msk = self._computemask(cc, Xi, Yi, Zi, bottomkey)
            Zi[msk] = np.nan

        self._set_section_bounds(cc, prop)
        return super(BaseSectionAxes, self).contourf(Xi, Yi, Zi, **kwargs)

    def pcolormesh(self, cc, prop, ninterp=30, sk=None, mask=True,
                   bottomkey="depth", interp_scheme="standard", **kwargs):
        if not isinstance(cc, AbstractCastCollection):
            raise TypeError("first argument must be a CastCollection type")

        (Xi, Yi, Zi) = self._interpolate_section(cc, prop, ninterp, interp_scheme)

        if sk is not None:
            Zi = self._smooth_field(Zi, sk)

        if mask:
            msk = self._computemask(cc, Xi, Yi, Zi, bottomkey)
            Zi[msk] = np.nan

        self._set_section_bounds(cc, prop)
        kwargs.setdefault("vmin", np.nanmin(Zi))
        kwargs.setdefault("vmax", np.nanmax(Zi))
        return super(BaseSectionAxes, self).pcolormesh(Xi, Yi, Zi, **kwargs)

    def add_bathymetry(self, bathymetry, vertices, maxdistance=200.0,
                       maxdepth=None, **kwargs):
        """ Add bathymetry from a Bathymetry object to a section plot.

        Arguments:
        ----------
        bathymetry      a Bathymetry2d instance
        vertices        a CastCollection, karta.Line, or list of points
                        defining a path

        Keyword arguments:
        ------------------
        ax              specific Axes to use
        maxdistance     the maximum distance a bathymetric observation
                        may be from a point in `vertices` to be plotted

        Additional keyword arguments are sent to `plt.fill_between`
        """
        if hasattr(vertices, "get_vertices"):
            vertices = vertices.get_vertices()
        elif hasattr(vertices, "coords"):
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
        plotbase = maxdepth*np.ones_like(depth)

        kwargs.setdefault("color", "0.0")
        kwargs.setdefault("zorder", 8)
        return self.fill_between(xalong, depth, plotbase, **kwargs)

class SectionAxes(BaseSectionAxes):
    """ Basic class for plotting an oceanographic section """
    name = "section"

    @staticmethod
    def _interpolate_section(cc, prop, ninterp, scheme="standard"):
        """ Scheme may be one of "standard", "horizontal_corr", or "zero_base" """
        Yo = np.vstack([cast[cast.primarykey] for cast in cc]).T
        Xo = np.tile(cc.projdist(), (len(Yo), 1))
        Zo = cc.asarray(prop)

        if scheme == "standard":
            msk = ~np.isnan(Xo+Yo+Zo)
            Xi, Yi = np.meshgrid(np.linspace(Xo[0,0], Xo[0,-1], ninterp),
                                 cc[0][cc[0].primarykey])
            Zi = griddata(np.c_[Xo[msk], Yo[msk]], Zo[msk],
                          np.c_[Xi.ravel(), Yi.ravel()], method="cubic")
            Zi = Zi.reshape(Xi.shape)

        elif scheme == "horizontal_corr":
            Xi, Yi = np.meshgrid(np.linspace(Xo[0,0], Xo[0,-1], ninterp),
                                 cc[0][cc[0].primarykey])

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

            Zi = griddata(np.c_[Xo.ravel(), Yo.ravel()], Zo.ravel(),
                          np.c_[Xi.ravel(), Yi.ravel()], method="linear")
            Zi = Zi.reshape(Xi.shape)

        elif scheme == "zero_base":

            Yo = np.vstack([cast[cast.primarykey] for cast in cc]).T
            Xo = np.tile(cc.projdist(), (len(Yo), 1))
            Zo = cc.asarray(prop)

            # Add zero boundary condition
            def _find_idepth(arr):
                idepth = []
                for col in arr.T:
                    _inonnan = np.arange(len(col))[~np.isnan(col)]
                    idepth.append(_inonnan[-1])
                return idepth

            xi = np.linspace(Xo[0,0], Xo[0,-1], ninterp)
            yi = cc[0][cc[0].primarykey]

            idxdepth = _find_idepth(Zo)
            idepth = np.round(np.interp(xi, Xo[0], idxdepth)).astype(int)

            xbc, ybc, zbc = [], [], []
            for j, i in enumerate(idepth):
                xbc.append(xi[j])
                ybc.append(yi[i]+2)
                zbc.append(0.0)

            msk = ~np.isnan(Xo + Yo + Zo)

            Xo_bc = np.r_[Xo[msk], xbc]
            Yo_bc = np.r_[Yo[msk], ybc]
            Zo_bc = np.r_[Zo[msk], zbc]

            Xi, Yi = np.meshgrid(xi, yi)

            alpha = 1e4
            Zi = griddata(np.c_[Xo_bc, alpha*Yo_bc], Zo_bc,
                          np.c_[Xi.ravel(), alpha*Yi.ravel()], method="cubic")

            # alpha = 1e2
            # rbfi = interpolate.Rbf(Xo_bc, alpha*Yo_bc, Zo_bc, function="thin_plate")
            # Zi = rbfi(Xi.ravel(), alpha*Yi.ravel())
            # Zi_check = rbfi(Xo.ravel(), alpha*Yo.ravel())

            Zi = Zi.reshape(Xi.shape)

        else:
            raise NotImplementedError("No scheme '{0}' implemented".format(scheme))
        return Xi, Yi, Zi

matplotlib.projections.register_projection(SectionAxes)

