"""
Bathymetry class that can be referenced to a CastCollection and automatically
plotted.
"""

from karta import Point, Line
from karta.vector.guppy import LONLAT

class Bathymetry(object):

    def __init__(self, x, y, z):
        self.line = Line(zip(x,y), data={"depth":z}, crs=LONLAT)
        return

    def atxy(self, x, y):
        """ Interpolate bottom depth at a point. """
        pt = Point((x, y))
        segments = tuple(self.line.segments())
        distances = [seg.shortest_distance_to(pt) for seg in segments]
        ii = distances.index(min(distances))
        a = segments[ii][0]
        b = segments[ii][1]
        ptonseg = segments[ii].nearest_on_boundary(pt)
        return (b.data["depth"] - a.data["depth"]) \
                * (a.greatcircle(ptonseg) / a.greatcircle(b)) \
                + a.data["depth"]




