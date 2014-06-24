"""
Bathymetry class that can be referenced to a CastCollection and automatically
plotted.
"""

from karta import Point, Line, LONLAT

class Bathymetry2d(Line):
    """ Bathymetric line
    Bathymetry2d(lon, lat, depth)
    """

    def __init__(self, vertices, depth=None, crs=LONLAT, **kw):
        kw.setdefault("crs", crs)
        if depth is not None:
            kw.update({"data": {"depth": depth}})
        else:
            depth = kw["data"]["depth"]
        super(Line, self).__init__(vertices, **kw)
        self.depth = depth
        return

    def atxy(self, x, y):
        """ Interpolate bottom depth at a point. """
        pt = Point((x, y), crs=LONLAT)
        segments = tuple(self.segments())
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


Bathymetry = Bathymetry2d

