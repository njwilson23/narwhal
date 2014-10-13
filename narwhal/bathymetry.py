"""
Bathymetry class that can be referenced to a CastCollection and automatically
plotted.
"""
import numpy as np
import karta
from karta import Point, Line

try:
    from karta.crs import crsreg
except ImportError:
    import karta as crsreg
LONLAT = crsreg.LONLAT
CARTESIAN = crsreg.CARTESIAN


class Bathymetry2d(Line):
    """ Bathymetric line
    Bathymetry2d(lon, lat, depth)
    """

    def __init__(self, vertices, depth=None, crs=LONLAT, **kw):
        kw.setdefault("crs", crs)
        if "data" not in kw:
            kw["data"] = {}
        if depth is None:
            if "depth" not in kw["data"]:
                raise KeyError("Bathymetry must be initialized with a 'depth' "
                               "argument or a dictionary with a 'depth' key")
            else:
                depth = kw["data"]["depth"]
        else:
            kw.update({"data": {"depth": depth}})
        super(Line, self).__init__(vertices, **kw)
        self.depth = np.asarray(depth)
        return

    def atxy(self, x, y):
        """ Interpolate bottom depth at a point. """
        pt = Point((x, y), crs=LONLAT)
        segments = tuple(self.segments)
        distances = [seg.shortest_distance_to(pt) for seg in segments]
        ii = distances.index(min(distances))
        a = segments[ii][0]
        b = segments[ii][1]
        ptonseg = segments[ii].nearest_on_boundary(pt)
        return (b.data["depth"] - a.data["depth"]) \
                * (a.greatcircle(ptonseg) / a.greatcircle(b)) \
                + a.data["depth"]

    def projdist(self, reverse=False):
        distances = [seg[0].greatcircle(seg[1]) for seg in self.segments()]
        cumulative = [0]
        for val in distances:
            cumulative.append(cumulative[-1] + val)
        if reverse:
            cumulative.reverse()
        return cumulative

    def project_along_cruise(self, cruiseline):
        """ Project depth locations to a cruise line.

        Returns:
        --------
        p       a vector of distances along the cruise
        q       a vector of distances from the cruise line
        """
        if self._crs != cruiseline._crs:
            raise karta.CRSError("CRS mismatch")

        P, Q = [], []
        for pt in self:
            npt = cruiseline.nearest_on_boundary(pt)
            p = 0.0
            for i, seg in enumerate(cruiseline.segments):
                if npt == seg.nearest_on_boundary(pt):
                    p += npt.distance(seg[0])
                    q = seg.shortest_distance_to(pt)
                    break
                else:
                    p += seg.length
            P.append(p)
            Q.append(q)
        return np.asarray(P), np.asarray(Q)

Bathymetry = Bathymetry2d

