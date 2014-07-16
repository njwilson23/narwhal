import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from scipy.interpolate import griddata, CloughTocher2DInterpolator
from scipy import ndimage
from narwhal import AbstractCast, AbstractCastCollection

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
    def _computemask(cc, Xi, Yi, Zi, bottomkey):
        depth = [cast.properties[bottomkey] for cast in cc]
        cx = cc.projdist()
        base = np.interp(Xi[0,:], cx, depth)
        zmask = Yi > np.tile(base, (Xi.shape[0], 1))
        return zmask

    def contour(self, cc, prop, ninterp=30, sk=None, mask=True,
                bottomkey="depth", **kwargs):

        if not isinstance(cc, AbstractCastCollection):
            raise TypeError("first argument must be a CastCollection type")

        (Xi, Yi, Zi) = self._interpolate_section(cc, prop, ninterp)

        if sk is not None:
            Zi = ndimage.filters.gaussian_filter(Zi, sk)

        if mask:
            msk = self._computemask(cc, Xi, Yi, Zi, bottomkey)
            Zi[msk] = np.nan

        self._set_section_bounds(cc, prop)
        return super(BaseSectionAxes, self).contour(Xi, Yi, Zi, **kwargs)

    def contourf(self, cc, prop, ninterp=30, sk=None, mask=True,
                 bottomkey="depth", **kwargs):

        if not isinstance(cc, AbstractCastCollection):
            raise TypeError("first argument must be a CastCollection type")

        (Xi, Yi, Zi) = self._interpolate_section(cc, prop, ninterp)

        if sk is not None:
            Zi = ndimage.filters.gaussian_filter(Zi, sk)

        if mask:
            msk = self._computemask(cc, Xi, Yi, Zi, bottomkey)
            Zi[msk] = np.nan

        self._set_section_bounds(cc, prop)
        return super(BaseSectionAxes, self).contourf(Xi, Yi, Zi, **kwargs)

    def pcolormesh(self, cc, prop, ninterp=30, sk=None, mask=True,
                   bottomkey="depth", **kwargs):
        if not isinstance(cc, AbstractCastCollection):
            raise TypeError("first argument must be a CastCollection type")

        (Xi, Yi, Zi) = self._interpolate_section(cc, prop, ninterp)

        if sk is not None:
            Zi = ndimage.filters.gaussian_filter(Zi, sk)

        if mask:
            msk = self._computemask(cc, Xi, Yi, Zi, bottomkey)
            Zi[msk] = np.nan

        self._set_section_bounds(cc, prop)
        kwargs.setdefault("vmin", np.nanmin(Zi))
        kwargs.setdefault("vmax", np.nanmax(Zi))
        return super(BaseSectionAxes, self).pcolormesh(Xi, Yi, Zi, **kwargs)

class SectionAxes(BaseSectionAxes):
    """ Basic class for plotting an oceanographic section """
    name = "section"

    @staticmethod
    def _interpolate_section(cc, prop, ninterp):
        Yo = np.vstack([cast[cast.primarykey] for cast in cc]).T
        Xo = np.tile(cc.projdist(), (len(Yo), 1))
        Zo = cc.asarray(prop)
        msk = ~np.isnan(Xo+Yo+Zo)
        Xi, Yi = np.meshgrid(np.linspace(Xo[0,0], Xo[0,-1], ninterp),
                             cc[0][cc[0].primarykey])
        Zi = griddata(np.c_[Xo[msk], Yo[msk]], Zo[msk],
                      np.c_[Xi.ravel(), Yi.ravel()], method="cubic")
        Zi = Zi.reshape(Xi.shape)
        print(Zi.shape)
        return Xi, Yi, Zi

class HorizontallyCorrelatedSectionAxes(BaseSectionAxes):
    """ SectionAxes where properties are assumed to be horizontally correlated
    over NaNs. """
    name = "horiz_corr_section"

    @staticmethod
    def _interpolate_section(cc, prop, ninterp):
        Yo = np.vstack([cast[cast.primarykey] for cast in cc]).T
        Xo = np.tile(cc.projdist(), (len(Yo), 1))
        Zo = cc.asarray(prop)
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
        return Xi, Yi, Zi

class ZeroBottomSectionAxes(BaseSectionAxes):
    """ SectionAxes where property is assumed to be zero at the maximum non-nan
    depth. """
    name = "zero_bottom_section"

    @staticmethod
    def _interpolate_section(cc, prop, ninterp):
        raise NotImplementedError()
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

        idepth = _find_idepth(Zo)
        idepth = np.round(np.interp(xi, Xo[0], idepth)).astype(int)

        xbc, ybc, zbc = [], [], []
        for j, i in enumerate(idepth):
            xbc.append(xi[j])
            ybc.append(yi[i])
            zbc.append(0.0)

        Xi, Yi = np.meshgrid(xi, yi)
        msk = ~np.isnan(Xo + Yo + Zo)

        Xo_bc = np.r_[Xo[msk], xbc]
        Yo_bc = np.r_[Yo[msk], ybc]
        Zo_bc = np.r_[Zo[msk], zbc]

        Zi = griddata(np.c_[Xo_bc, 1e4*Yo_bc], Zo_bc,
                      np.c_[Xi.ravel(), 1e4*Yi.ravel()], method="linear")
        Zi = Zi.reshape(Xi.shape)
        # Zi[~msk] = np.nan

        return Xi, Yi, Zi

matplotlib.projections.register_projection(SectionAxes)
matplotlib.projections.register_projection(HorizontallyCorrelatedSectionAxes)
matplotlib.projections.register_projection(ZeroBottomSectionAxes)

