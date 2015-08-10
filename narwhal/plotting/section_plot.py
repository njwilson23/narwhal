import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# import matplotlib.tri as mtri
from scipy.interpolate import griddata, CloughTocher2DInterpolator
from scipy import ndimage
from karta import Line
from karta.crs import LonLatWGS84
from narwhal import AbstractCastCollection

class BaseSectionAxes(plt.Axes):

    def __init__(self, *args, **kwargs):
        """ ykey        Key in casts to use along y-axis """
        self.ykey = kwargs.pop("ykey", "depth")
        super(plt.Axes, self).__init__(*args, **kwargs)

    def _set_section_bounds(self, cc, prop):
        """ Given a CastCollection and a property name, set the axes bounds to
        propertly show all valid data. """
        # vertical extents
        zgen = (c[self.ykey].values for c in cc)
        validmsk = (~np.isnan(c[prop].values) for c in cc)
        ymax = max(z[msk][-1] for z,msk in zip(zgen, validmsk))
        self.set_ylim((ymax, 0))

        # horizontal extents
        cx = cc.projdist()
        self.set_xlim((cx[0], cx[-1]))
        return

    @staticmethod
    def _smooth_field(d, sk):
        msk = np.isnan(d)
        d[msk] = 0.0
        d_ = ndimage.filters.gaussian_filter(d, sk)
        d_[msk] = np.nan
        return d_

    def _computemask(self, cc, Xi, Yi, Zi, bottomkey):
        if bottomkey is not None:
            depth = [cast.properties[bottomkey] for cast in cc]
        else:
            def _last(arr):
                msk = ~np.isnan(arr)
                return arr[np.max(np.argwhere(msk))]
            depth = [_last(cast[self.ykey].values) for cast in cc]
        cx = cc.projdist()
        base = np.interp(Xi[0,:], cx, depth)
        zmask = Yi > np.tile(base, (Xi.shape[0], 1))
        return zmask

    def hatch(self, nx=20, **kw):
        """ Add a hatch pattern to section to represent NaNs. *nx* gives the
        number of hatch lines to draw. """
        ny = nx * self.bbox.height / self.bbox.width
        x0, x1 = self.get_xlim()
        y0, y1 = self.get_ylim()
        dx = (x1-x0) / nx
        m = (nx/ny) * abs((y1-y0)/(x1-x0))

        LXbottom = np.linspace(x0 - abs(y1-y0)/m, x1-dx, nx)
        LXtop = LXbottom + abs(y1-y0)/m
        LX = np.empty(3*nx, dtype=np.float64)
        LY = np.empty(3*nx, dtype=np.float64)
        LX[0::3] = LXbottom
        LX[1::3] = LXtop
        LX[2::3] = np.nan
        LY[0::3] = y0
        LY[1::3] = y1
        LY[2::3] = np.nan
        kw.setdefault("color", "black")
        kw.setdefault("lw", 0.4)
        kw.setdefault("zorder", -1)
        self.plot(LX, LY, **kw)
        return

    def hatch_(self, hatch="/"):
        """ Add a hatch pattern to section to represent NaNs. """
        x0, x1 = self.get_xlim()
        y0, y1 = self.get_ylim()
        xy = (x0, y0)
        width = x1 - x0
        height = y1 - y0
        bg = matplotlib.patches.Rectangle(xy, width, height, hatch=hatch,
                fill=False, zorder=-1)
        self.add_patch(bg)
        return bg

    def contour(self, cc, prop, ninterp=30, sk=None, mask=True,
                bottomkey="depth", interp_scheme="horizontal_corr", **kwargs):

        if not isinstance(cc, AbstractCastCollection):
            raise TypeError("first argument must be a CastCollection type")

        (Xi, Yi, Zi) = self._interpolate_section(cc, prop, ninterp, interp_scheme)

        if sk is not None:
            Zi = self._smooth_field(Zi, sk)

        if mask:
            msk = self._computemask(cc, Xi, Yi, Zi, bottomkey)
            Zi[msk] = np.nan

        self._set_section_bounds(self.ykey, cc, prop)
        return super(BaseSectionAxes, self).contour(Xi, Yi, Zi, **kwargs)

    def contourf(self, cc, prop, ninterp=30, sk=None, mask=True,
                 bottomkey="depth", interp_scheme="horizontal_corr", **kwargs):

        if not isinstance(cc, AbstractCastCollection):
            raise TypeError("first argument must be a CastCollection type")

        (Xi, Yi, Zi) = self._interpolate_section(cc, prop, ninterp, interp_scheme)

        if sk is not None:
            Zi = self._smooth_field(Zi, sk)

        if mask:
            msk = self._computemask(cc, Xi, Yi, Zi, bottomkey)
            Zi[msk] = np.nan

        self._set_section_bounds(self.ykey, cc, prop)
        return super(BaseSectionAxes, self).contourf(Xi, Yi, Zi, **kwargs)

    def pcolormesh(self, cc, prop, ninterp=30, sk=None, mask=True,
                   bottomkey="depth", interp_scheme="horizontal_corr", **kwargs):
        if not isinstance(cc, AbstractCastCollection):
            raise TypeError("first argument must be a CastCollection type")

        (Xi, Yi, Zi) = self._interpolate_section(cc, prop, ninterp, interp_scheme)

        if sk is not None:
            Zi = self._smooth_field(Zi, sk)

        if mask:
            msk = self._computemask(cc, Xi, Yi, Zi, bottomkey)
            Zi[msk] = np.nan

        self._set_section_bounds(self.ykey, cc, prop)
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

        cruiseline = Line(vertices, crs=LonLatWGS84)
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

    def mark_stations(self, cc, **kwargs):
        """ Draw a vertical line at each station position along the section.
        Keyword arguments are passed to `self.plot` """
        ymax = max(np.nanmax(np.array(c[self.ykey])) for c in cc)
        kwargs.setdefault("color", "0.3")
        kwargs.setdefault("linestyle", "--")
        lines = [self.plot((x_, x_), (ymax, 0), **kwargs) for x_ in cc.projdist()]
        return lines

    def label_stations(self, cc, labels, vert_offset=20, **kwargs):
        """ Add labels for each station in the section at the top of the plot.

        Arguments
        ---------
        cc              CastCollection used to draw section

        labels          one of:
                          - a list of strings where `len(labels) == len(cc)`
                          - a string matching a key in Cast.properties
                          - a function that takes a cast and returns a string

        vert_offset     sets the vertical position of each label

        Keyword arguments are passed to `self.text`
        """
        if isinstance(labels, str):
            labels_str = (c.properties[labels] for c in cc)
        elif hasattr(labels, "__call__"):
            labels_str = (labels(c) for c in cc)
        else:
            labels_str = labels
        cx = cc.projdist()
        texts = []
        kwargs.setdefault("ha", "center")
        kwargs.setdefault("size", plt.rcParams["font.size"]-2)
        for x, c, label in zip(cx, cc, labels_str):
            txt = self.text(x, -vert_offset, label, **kwargs)
            texts.append(txt)
        return texts


class SectionAxes(BaseSectionAxes):
    """ Basic class for plotting an oceanographic section """
    name = "section"

    def _interpolate_section(self, cc, prop, ninterp, scheme="horizontal_corr"):
        """ Scheme may be one of "cubic", "horizontal_corr", or "zero_base" """

        def _longest_cast(cc):
            """ Return the longest cast in a cast collection """
            max_y = max(np.nanmax(c[self.ykey]) for c in cc)
            for cast in cc:
                if max_y in cast[self.ykey]:
                    return cast

        if scheme == "cubic":
            Yo = np.vstack([cast[self.ykey] for cast in cc]).T
            Xo = np.tile(cc.projdist(), (len(Yo), 1))
            Zo = cc.asarray(prop)

            msk = ~np.isnan(Xo+Yo+Zo)
            c = _longest_cast(cc)
            longest_z = c[self.ykey]

            Xi, Yi = np.meshgrid(np.linspace(Xo[0,0], Xo[0,-1], ninterp), longest_z)
            Zi = griddata(np.c_[Xo[msk], Yo[msk]], Zo[msk],
                          np.c_[Xi.ravel(), Yi.ravel()], method="cubic")
            Zi = Zi.reshape(Xi.shape)

        elif scheme == "horizontal_corr":

            # this version works with irregular gridding

            X, Y = [], []
            Z = []

            distances = cc.projdist()
            castleft, xleft = None, None

            for i,x in enumerate(distances):
                cast = cc[i]
                if i == len(cc) - 1:
                    castrigh = None
                    xrigh = None
                else:
                    castrigh = cc[i+1]
                    xrigh = distances[i+1]
                    
                # add the measured values
                X.extend(x * np.ones(cast.nvalid(prop)))
                Y.extend(cast[self.ykey][~cast.nanmask(prop)])
                Z.extend(cast[prop][~cast.nanmask(prop)])
                    
                # for each non-NaN level in *cast*, check castleft and castrigh
                # to see if they're NaN
                # if so, add a dummy value half way between
                msk = ~cast.nanmask(prop)
                for lvl, v in zip(cast[self.ykey][msk], cast[prop][msk]):

                    if castleft and \
                            np.isnan(castleft.interpolate(prop, self.ykey, lvl)):
                        X.append(0.5 * (xleft+x))
                        Y.append(lvl)
                        Z.append(v)
                
                    if castrigh and \
                            np.isnan(castrigh.interpolate(prop, self.ykey, lvl)):
                        X.append(0.5 * (xrigh+x))
                        Y.append(lvl)
                        Z.append(v)

                castleft = cast
                xleft = x

            X = np.array(X)
            Y = np.array(Y)

            max_y = max(np.nanmax(c[self.ykey].values) for c in cc)
            max_x = distances[-1]
            y_int = int(round(min(5, max_y/100)))
            x_int = max_x / ninterp
            Yi, Xi = np.mgrid[0:max_y+0.1*y_int:y_int, 0:max_x+0.1*x_int:x_int]
            ct = CloughTocher2DInterpolator(np.c_[0.0001*X,Y], Z)
            Zi = ct(0.0001*Xi, Yi)

        elif scheme == "zero_base":

            Yo = np.vstack([cast[self.ykey] for cast in cc]).T
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
            c = _longest_cast(cc)
            yi = c[self.ykey]

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

