"""
Bathymetry class that can be referenced to a CastCollection and automatically
plotted.
"""

from karta import Point, Line

class Bathymetry(object):

    def __init__(self, x, y, z):
        self.line = Line(zip(x,y), data={"depth":z})
        return

    def atxy(self, x, y):
        """ Interpolate bottom depth at a point. """
        pt = self.line.nearest_on_boundary(Point(x, y))
        #da = self.line.shortest_distance_to(pt)

        return




