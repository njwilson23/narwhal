"""
Bathymetry class that can be referenced to a CastCollection.

Purposes of the Bathymetry type:
    - plotting sections
    - storing point bathymetric data (a shapefile may be a better solution)
"""
import numpy as np
import scipy.interpolate

try:
    from karta import Line
    from karta.crs import LonLatWGS84
except ImportError:
    # Frankly, this is unlikely to work right now
    from .geo import Line, LonLatWGS84

class Bathymetry(object):
    """ Class to represent bathymetric observations, either along a line or at
    a scattering of points. The observation geometry is given by a geometry
    object, which is expected to be an instance of Line or Multipoint from
    either `karta.vector.geometry` or `narwhal.geo`
    """

    def __init__(self, depth, geo):
        depth = list(depth)
        if len(geo) != len(depth):
            raise ValueError("length of depth array and geometry must match")
        self.geo = geo
        self.depth = depth

        # Naive implementation of a function that return a distance matrix based
        # on a geographical distance norm
        def geo_norm(xy1, xy2):
            x1 = xy1[0].ravel()
            y1 = xy1[1].ravel()
            x2 = xy2[0].ravel()
            y2 = xy2[1].ravel()
            return np.array([geo.crs.inverse(x1_, y1_, x2_, y2_)[2]
                             for x1_, y1_ in zip(x1.ravel(), y1.ravel())
                             for x2_, y2_ in zip(x2.ravel(), y2.ravel())]).reshape([len(x1), len(x2)])
        self.interpolator = scipy.interpolate.Rbf([pt.x for pt in geo],
                                                  [pt.y for pt in geo],
                                                  depth,
                                                  norm=geo_norm)
        return

    def atpoint(self, point):
        """ Return the depth interpolated at Point *point* using `self.interpolator` """
        if point.crs != self.geo.crs:
            x, y = self.geo.crs.project(*point.crs.project(point.x, point.y, inverse=True))
        else:
            x, y = point.x, point.y
        return float(self.interpolator(x, y))

    def projdist(self):
        """ Return the cumulative distance along a linear Bathymetry. """
        idx = range(1, len(self.geo))
        d = [0.0]
        for i in idx:
            d.append(d[-1] + self.geo[i].distance(self.geo[i-1]))
        return d

    def project_along_cruise(self, cruiseline):
        raise NotImplementedError()

class Bathymetry2d(Line):
    """ Bathymetric line
    Bathymetry2d(lon, lat, depth)

    WARNING: this class is DEPRECATED in favor of narwhal.bathymetry.Bathymetry
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
                              crs=self.crs, copy_metadata=False)
        return subset

    def atpoint(self, pt):
        """ Return bottom depth at a point by finding the point on the
        cruiseline nearest to *pt* and interpolating linearly between the depth
        measurements on either side.
        """
        segments = tuple(self.segments)
        distances = [seg.shortest_distance_to(pt) for seg in segments]
        ii = distances.index(min(distances))
        (a, b) = segments[ii]
        ptonseg = segments[ii].nearest_on_boundary(pt)
        adepth = a.data[0][a.data.fields.index("depth")]
        bdepth = b.data[0][b.data.fields.index("depth")]
        return (bdepth-adepth) * (a.distance(ptonseg)/a.distance(b)) + adepth

    def projdist(self, reverse=False):
        distances = [seg[0].distance(seg[1]) for seg in self.segments()]
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
        if self.crs != cruiseline.crs:
            raise TypeError("CRS mismatch")

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
