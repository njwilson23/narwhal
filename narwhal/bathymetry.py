"""
Bathymetry class that can be referenced to a CastCollection and automatically
plotted.
"""
import numpy as np
import karta
from karta import Point, Line
from karta.crs import LonLatWGS84

class Bathymetry2d(Line):
    """ Bathymetric line
    Bathymetry2d(lon, lat, depth)
    """

    def __init__(self, vertices, depth=None, crs=LonLatWGS84, **kw):
        kw.setdefault("data", {})
        vertices = list(vertices)
        if len(depth) != len(vertices):
            raise ValueError("Depth array length must matches vertices "
                             "length")
        kw["data"]["depth"] = depth
        super(Line, self).__init__(vertices, crs=crs, **kw)
        self.depth = np.asarray(depth)
        return

    def _subset(self, idxs):
        """ Return a subset defined by index in *idxs*. """
        vertices = [self.vertices[i] for i in idxs]
        depth = [self.depth[i] for i in idxs]
        subset = Bathymetry2d(vertices, depth=depth, 
                              crs=self._crs, copy_metadata=False)
        return subset

    def atxy(self, x, y):
        """ Interpolate bottom depth at a point. """
        pt = Point((x, y), crs=LonLatWGS84)
        segments = tuple(self.segments)
        distances = [seg.shortest_distance_to(pt) for seg in segments]
        ii = distances.index(min(distances))
        (a, b) = segments[ii]
        ptonseg = segments[ii].nearest_on_boundary(pt)
        adepth = a.data[0][a.data.fields.index("depth")]
        bdepth = b.data[0][b.data.fields.index("depth")]
        return (bdepth-adepth) * (a.distance(ptonseg)/a.distance(b)) + adepth

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

