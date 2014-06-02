"""
Bathymetry class that can be referenced to a CastCollection and automatically
plotted.
"""

from karta import Point, Line, LONLAT

class Bathymetry2d(object):

    def __init__(self, x, y, z):
        if len(x) != len(y) != len(z):
            raise ValueError("x, y, z must all have the same length")
        self.line = Line(zip(x,y), data={"depth":z}, crs=LONLAT)
        self.depth = z
        return

    def atxy(self, x, y):
        """ Interpolate bottom depth at a point. """
        pt = Point((x, y), crs=LONLAT)
        segments = tuple(self.line.segments())
        distances = [seg.shortest_distance_to(pt) for seg in segments]
        ii = distances.index(min(distances))
        a = segments[ii][0]
        b = segments[ii][1]
        ptonseg = segments[ii].nearest_on_boundary(pt)
        return (b.data["depth"] - a.data["depth"]) \
                * (a.greatcircle(ptonseg) / a.greatcircle(b)) \
                + a.data["depth"]

    def projdist(self, reverse=False):
        distances = [seg[0].greatcircle(seg[1]) for seg in self.line.segments()]
        cumulative = [0]
        for val in distances:
            cumulative.append(cumulative[-1] + val)
        if reverse:
            cumulative.reverse()
        return cumulative


Bathymetry = Bathymetry2d

