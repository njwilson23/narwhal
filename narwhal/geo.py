""" Define some geometrical classes with an interface similar to karta
(http://ironicmtn.com/karta.html)

If karta is not available, this module will be loaded instead and provides some
bare-bones geometrical and geodetic capabilities.
"""

from __future__ import print_function, division
from math import sqrt, sin, cos, tan, atan, atan2, pi

class CoordinateSystem(object):
    pass


class LonLat(CoordinateSystem):

    def __init__(self, a, b):
        """ Define a geographical coordinate system on an ellipse. *a* is the
        equatorial radius, and *b* is the polar radius. """
        self.a = a
        self.b = b
        return

    def get_proj4(self):
        return "+proj=longlat +units=m +no_defs +a={a} +b={b}".format(a=self.a, b=self.b)

    def project(self, x, y, **kw):
        """ Projection on a geographical coordinate system is the identity """
        return x, y

    def forward(self, x, y, azimuth, distance):
        """ Compute the final position after moving from (x, y) in the *azimuth*
        direction for *distance* meters

        Algorithm due to Karney, C.F.F. "Algorithms for geodesics", J. Geod (2013)
        """
        f = (self.a-self.b) / self.a

        phi1 = pi*y/180.0
        alpha1 = pi*azimuth/180.0

        # Solve triangle NEA from Karney Fig. 1
        beta1 = atan((1-f)*tan(phi1))
        _i = sqrt(cos(alpha1)**2 + (sin(alpha1)*sin(beta1))**2)
        alpha0 = atan2(sin(alpha1)*cos(beta1), _i)
        sigma1 = atan2(sin(beta1), cos(alpha1)*cos(beta1))
        omega1 = atan2(sin(alpha0)*sin(sigma1), cos(sigma1))

        # Determine sigma2
        eps2 = (f*(2-f))
        eps = sqrt(eps2)
        second_eps = sqrt(eps2 / (1-eps2))
        k2 = (second_eps * cos(alpha0))**2

        _rad = sqrt(1+k2)
        eps = (_rad - 1) / (_rad + 1)
        A1 = 1.0/(1-eps) * (1 + eps**2/4 + eps**4/64 + eps**6/256)
        C1 = [-1.0/2*eps + 3.0/16*eps**3 - 1.0/32*eps**5,
              -1.0/16*eps**2 + 1.0/32*eps**4 - 9.0/2048*eps**6,
              -1.0/48*eps**3 + 3.0/256*eps**5,
              -5.0/512*eps**4 + 3.0/512*eps**6,
              -7.0/1280*eps**5,
              -7.0/2048*eps**6]

        I1 = A1 * (sigma1 + sum(c*sin(2*(i+1)*sigma1) for i,c in enumerate(C1)))
        s1 = I1 * self.b
        s2 = s1 + distance
        tau2 = s2 / (self.b*A1)

        C1p = [eps/2 - 9.0/32*eps**3 + 205.0/1536*eps**5,
               5.0/16*eps**2 - 37.0/96*eps**4 + 1335.0/4096*eps**6,
               29.0/96*eps**3 - 75.0/128*eps**5,
               539.0/1536*eps**4 - 2391.0/2560*eps**6,
               3467.0/7680*eps**5,
               38081.0/61440*eps**6]

        sigma2 = tau2 + sum(c*sin(2*(i+1)*tau2) for i,c in enumerate(C1p))

        # Solve triangle NEB in Karney Fig. 1
        alpha2 = atan2(sin(alpha0), cos(alpha0)*cos(sigma2))
        _j = sqrt((cos(alpha0)*cos(sigma2))**2 + sin(alpha0)**2)
        beta2 = atan2(cos(alpha0)*sin(sigma2), _j)
        omega2 = atan2(sin(alpha0)*sin(sigma2), cos(sigma2))

        # Determine lambda12
        n = f / (2.0-f)
        n2 = n*n
        A3 = 1.0 - (1.0/2-n/2)*eps - (1.0/4 + n/8 - 3.0*n2/8)*eps**2 \
            - (1.0/16 + 3.0*n/16 + n2/16)*eps**3 - (3.0/64 + n/32)*eps**4 \
            - 3.0/128*eps**5

        C3 = [(1.0/4 - n/4)*eps + (1.0/8 - n2/8)*eps**2 + (3.0/64 + 3.0*n/64 - n2/64)*eps**3 \
                + (5.0/128 + n/64)*eps**4 + 3.0/128*eps**5,
              (1.0/16 - 3.0*n/32 + n2/32)*eps**2 + (3.0/64 - n/32 - 3*n2/64)*eps**3 \
                + (3.0/128 + n/128)*eps**4 + 5.0/256*eps**5,
              (5.0/192 - 3.0*n/64 + 5.0*n2/192)*eps**3 + (3.0/128 - 5.0*n/192)*eps**4 \
                + 7.0/512*eps**5,
              (7.0/512 - 7.0*n/256)*eps**4 + 7.0*eps**5/512,
              21.0*eps**5/2560]

        I3s1 = A3 * (sigma1 + sum(c*sin(2*(i+1)*sigma1) for i,c in enumerate(C3)))
        I3s2 = A3 * (sigma2 + sum(c*sin(2*(i+1)*sigma2) for i,c in enumerate(C3)))

        lambda1 = omega1 - f*sin(alpha0)*I3s1
        lambda2 = omega2 - f*sin(alpha0)*I3s2
        lambda12 = lambda2 - lambda1

        phi2 = atan(tan(beta2) / (1-f))
        x2 = x + lambda12*180.0/pi
        if x2 >= 180.0:
            x2 -= 360.0
        y2 = phi2*180.0/pi
        return x2, y2

    def inverse(self, x0, y0, x1, y1):
        """ Compute the shortest path (geodesic) between two points """

        backaz = az + pi
        return az, backaz, distance

def _flattening(a, b):
    return (a-b)/a

def _degrees(r):
    return r*180/pi

def printd(*args):
    argsd = [_degrees(a) for a in args]
    print(*argsd)

LonLatWGS84 = LonLat(6378137.0, 6356752.314245)

class Point(object):
    _geotype = "Point"

    def __init__(vertex, crs=LonLatWGS84):
        self.vertex = vertex
        self.crs = crs

class MultipointBase(object):

    def __init__(vertices, crs=LonLatWGS84):
        self.vertices = vertices
        self.crs = crs

    @property
    def coordinates(self):
        return list(*zip(*self.vertices))

class Multipoint(MultipointBase):
    _geotype = "Multipoint"

class Line(MultipointBase):
    _geotype = "Line"

if __name__ == "__main__":
    # Karney's Table 2 example for the forward problem
    x1, y1 = LonLatWGS84.forward(0.0, 40.0, 30.0, 10e6)
    print("Solution:", 137.844, 41.793)
    print(x1, y1)
