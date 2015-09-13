""" Define some geometrical classes with an interface similar to karta
(http://ironicmtn.com/karta.html)

If karta is not available, this module will be loaded instead and provides some
bare-bones geometrical and geodetic capabilities.
"""

from __future__ import print_function, division
import warnings
from math import sqrt, sin, cos, tan, asin, acos, atan, atan2, pi
from scipy.optimize import brentq

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
        """ Compute the destination reached starting from a point and travelling
        in a specified direction.

        Parameters
        ----------
        x:          longitude at start
        y:          latitude at start
        azimuth:    direction travelled from point
        distnce:    distance travelled

        Returns
        -------
        x2:         longitude at destination
        y2:         latitude at destination
        back_az:    back azimuth from destination

        Algorithm due to Karney, C.F.F. "Algorithms for geodesics", J. Geod (2013)
        """
        f = (self.a-self.b) / self.a

        phi1 = pi*y/180.0
        alpha1 = pi*azimuth/180.0

        # Solve triangle NEA from Karney Fig. 1
        beta1 = atan((1-f)*tan(phi1))
        _i = sqrt(cos(alpha1)**2 + (sin(alpha1)*sin(beta1))**2)
        alpha0 = atan2(sin(alpha1)*cos(beta1), _i)
        # sigma1 = atan2(sin(beta1), cos(alpha1)*cos(beta1))
        # omega1 = atan2(sin(alpha0)*sin(sigma1), cos(sigma1))
        sigma1, omega1 = _solve_NEA(alpha0, alpha1, beta1)

        # Determine sigma2
        eccn2 = (f*(2-f))
        second_eccn2 = eccn2 / (1-eccn2)
        k2 = second_eccn2*cos(alpha0)**2

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
        backaz = (alpha2+pi)*180/pi
        x2 = (x2+180) % 360 - 180
        backaz = (backaz+180) % 360 - 180
        return x2, y2, backaz

    def _inverse_equatorial(self, x1, x2):
        if x1 - x2 > 0:
            az = 90.0
            baz = 270.0
        else:
            ax = 270.0
            baz = 90.0
        s12 = 2 * pi * self.a * abs(x1-x2)/360.0
        return az, baz, s12

    def inverse(self, x1, y1, x2, y2, tol=None):
        """ Compute the shortest path (geodesic) between two points.

        Parameters
        ----------
        x1:         first longitude
        y1:         first latitude
        x2:         second longitude
        y2:         second latitude

        Returns
        -------
        az:         forward azimuth from first point
        back_az:    back azimuth from second point
        distance:   distance between points

        Algorithm due to Karney, C.F.F. "Algorithms for geodesics", J. Geod (2013)
        """
        niter = 0
        maxiter = 100
        if tol is None:
            tol = 1e-10

        if y1 == y2 == 0:
            # Equatorial case
            return self._inverse_equatorial(x1, x2)

        # Canonical configuration
        tr, x1, y1, x2, y2 = _canonical_configuration(x1, y1, x2, y2)

        phi1 = y1*pi/180.0
        phi2 = y2*pi/180.0
        lambda12 = (x2-x1)*pi/180.0
        f = (self.a-self.b) / self.a

        # Guess the azimuth
        if (abs(lambda12-pi) > 0.0087) or (abs(phi1+phi2) > 0.0087):
            # not nearly antipodal
            alpha1, _, _ = solve_vicenty(self.a, f, lambda12, phi1, phi2)
        else:
            alpha1 = solve_astroid(self.a, f, lambda12, phi1, phi2)

        beta1 = atan((1-f)*tan(phi1))
        beta2 = atan((1-f)*tan(phi2))

        eccn2 = f*(2-f)
        second_eccn2 = eccn2 / (1-eccn2)

        if x1 == x2:
            # Meridional case 1
            alpha0 = alpha1 = alpha2 = omega1 = omega2 = 0.0

            _i = sqrt(cos(alpha1)**2 + (sin(alpha1)*sin(beta1))**2)
            alpha0 = atan2(sin(alpha1)*cos(beta1), _i)
            sigma1, _ = _solve_NEA(alpha0, alpha1, beta1)
            _, sigma2, _ = _solve_NEB(alpha0, alpha1, beta1, beta2)

            k2 = second_eccn2
            _rad = sqrt(1+k2)
            eps = (_rad - 1) / (_rad + 1)

        elif abs(lambda12 % (2*pi) - pi) < 1e-12:
            # Meridional case 2
            if y1 + y2 > 0:
                alpha0 = alpha1 = 0.0
                alpha2 = omega1 = omega2 = pi
            else:
                alpha0 = alpha1 = omega1 = omega2 = pi
                alpha2 = 0.0

            sigma1, _ = _solve_NEA(alpha0, alpha1, beta1)
            _, sigma2, _ = _solve_NEB(alpha0, alpha1, beta1, beta2)

            k2 = second_eccn2
            _rad = sqrt(1+k2)
            eps = (_rad - 1) / (_rad + 1)

        else:
            # Newton iteration
            dlambda12 = tol + 1

            while (abs(dlambda12) > tol) and (niter != maxiter):

                # Solve triangles
                _i = sqrt(cos(alpha1)**2 + (sin(alpha1)*sin(beta1))**2)
                alpha0 = atan2(sin(alpha1)*cos(beta1), _i)
                sigma1, omega1 = _solve_NEA(alpha0, alpha1, beta1)
                alpha2, sigma2, omega2 = _solve_NEB(alpha0, alpha1, beta1, beta2)

                # Determine lambda12
                k2 = second_eccn2 * cos(alpha0)**2
                _rad = sqrt(1+k2)
                eps = (_rad - 1) / (_rad + 1)

                n = f/(2-f)
                n2 = n*n
                A3 = 1.0 - (1.0/2 - 1.0/2*n)*eps - (1.0/4 + 1.0/8*n - 3.0/8*n2)*eps**2 \
                    - (1.0/16 + 3.0/16*n + 1.0/16*n2)*eps**3 - (3.0/64 + 1.0/32*n)*eps**4 \
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
                lambda12_next = lambda2 - lambda1
                dlambda12 = lambda12_next - lambda12

                if abs(dlambda12) > tol:
                    # Refine alpha1
                    A1 = 1.0/(1-eps) * (1 + eps**2/4 + eps**4/64 + eps**6/256)
                    C1 = [-1.0/2*eps + 3.0/16*eps**3 - 1.0/32*eps**5,
                          -1.0/16*eps**2 + 1.0/32*eps**4 - 9.0/2048*eps**6,
                          -1.0/48*eps**3 + 3.0/256*eps**5,
                          -5.0/512*eps**4 + 3.0/512*eps**6,
                          -7.0/1280*eps**5,
                          -7.0/2048*eps**6]

                    I1s1 = A1 * (sigma1 + sum(c*sin(2*(i+1)*sigma1) for i,c in enumerate(C1)))
                    I1s2 = A1 * (sigma2 + sum(c*sin(2*(i+1)*sigma2) for i,c in enumerate(C1)))

                    A2 = (1-eps) * (1 + 1.0/4*eps**2 + 9.0/64*eps**4 + 25.0/256*eps**6)
                    C2 = [1.0/2*eps + 1.0/16*eps**3 + 1.0/32*eps**5,
                          3.0/16*eps**2 + 1.0/32*eps**4 + 35.0/2048*eps**6,
                          5.0/48*eps**3 + 5.0/256*eps**5,
                          35.0/512*eps**4 + 7.0/512*eps**6,
                          63.0/1280*eps**5,
                          77.0/2048*eps**6]

                    I2s1 = A2 * (sigma1 + sum(c*sin(2*(i+1)*sigma1) for i,c in enumerate(C2)))
                    I2s2 = A2 * (sigma2 + sum(c*sin(2*(i+1)*sigma2) for i,c in enumerate(C2)))

                    Js1 = I1s1 - I2s1
                    Js2 = I1s2 - I2s2

                    m12 = self.b * (sqrt(1 + k2*sin(sigma2)**2) * cos(sigma1)*sin(sigma2) \
                                  - sqrt(1 + k2*sin(sigma1)**2) * sin(sigma1)*cos(sigma2) \
                                  - cos(sigma1) * cos(sigma2) * (Js2-Js1))
                    dlambda12_dalpha1 = m12/(self.a * cos(alpha2)*cos(beta2))
                    dalpha1 = -dlambda12 / dlambda12_dalpha1
                    alpha1 = (alpha1 + dalpha1) % (2*pi)

                niter += 1

        if niter == maxiter:
            warnings.warn("Convergence failure", warnings.RuntimeWarning)

        k2 = second_eccn2 * cos(alpha0)**2
        _rad = sqrt(1+k2)
        eps = (_rad - 1) / (_rad + 1)

        # Determine s12
        A1 = 1.0/(1-eps) * (1 + eps**2/4 + eps**4/64 + eps**6/256)
        C1 = [-1.0/2*eps + 3.0/16*eps**3 - 1.0/32*eps**5,
              -1.0/16*eps**2 + 1.0/32*eps**4 - 9.0/2048*eps**6,
              -1.0/48*eps**3 + 3.0/256*eps**5,
              -5.0/512*eps**4 + 3.0/512*eps**6,
              -7.0/1280*eps**5,
              -7.0/2048*eps**6]

        I1s1 = A1 * (sigma1 + sum(c*sin(2*(i+1)*sigma1) for i,c in enumerate(C1)))
        I1s2 = A1 * (sigma2 + sum(c*sin(2*(i+1)*sigma2) for i,c in enumerate(C1)))

        s1 = I1s1*self.b
        s2 = I1s2*self.b
        s12 = s2-s1

        if tr["xflip"]:
            alpha1 = -alpha1
            alpha2 = -alpha2

        if tr["yflip"]:
            alpha1, alpha2 = pi-alpha2, pi-alpha1

        if tr["ysignswap"]:
            alpha1 = pi - alpha1
            alpha2 = pi - alpha2

        az = alpha1*180/pi
        backaz = (alpha2+pi)*180/pi
        return az % (360), backaz % (360), s12

def _normalize_longitude(x):
    """ Return longitude in the range [-180, 180). """
    return (x+180) % 360 - 180

def _canonical_configuration(x1, y1, x2, y2):
    """ Put coordinates into a configuration where (Karney, eqn 44)
        y1 <= 0
        y1 <= y2 <= -y1
        0 <= x2-x1 <= 180
    """
    transformation = dict(yflip=False, xflip=False, ysignswap=False)

    if abs(y1) < abs(y2):
        y1, y2 = y2, y1
        transformation["yflip"] = True

    if y1 > 0:
        y1, y2 = -y1, -y2
        transformation["ysignswap"] = True

    x2 = _normalize_longitude(x2-x1)
    x1 = 0.0

    if (x2 < 0) or (x2 > 180):
        x2 = -x2
        transformation["xflip"] = True

    return transformation, x1, y1, x2, y2

def _solve_NEA(alpha0, alpha1, beta1):
    sigma1 = atan2(sin(beta1), cos(alpha1)*cos(beta1))
    omega1 = atan2(sin(alpha0)*sin(sigma1), cos(sigma1))
    return sigma1, omega1

def _solve_NEB(alpha0, alpha1, beta1, beta2):
    try:
        alpha2 = acos(sqrt(cos(alpha1)**2*cos(beta1)**2 + (cos(beta2)**2 - cos(beta1)**2)) / cos(beta2))
    except ValueError:
        alpha2 = asin(sin(alpha0) / cos(beta2))     # Less accurate?
    sigma2 = atan2(sin(beta2), cos(alpha2)*cos(beta2))
    omega2 = atan2(sin(alpha0)*sin(sigma2), cos(sigma2))
    return alpha2, sigma2, omega2

def solve_astroid(a, f, lambda12, phi1, phi2):
    """ Used to provide an initial guess to the inverse problem by solving the
    corresponding problem on a sphere.

    Parameters
    ----------
    a:          equatorial radius
    f:          flattening
    lambda12:   difference in longitudes (radians)
    phi1:       first latitude (radians)
    phi2:       second latitude (radians)

    Returns
    -------
    alpha1:     estimated forward azimuth at first point

    see Karney (2013) J. Geod. for details
    """
    beta1 = atan((1-f) * tan(phi1))
    beta2 = atan((1-f) * tan(phi2))
    delta = f*a*pi*cos(beta1)**2
    x = (lambda12-pi) * (a*cos(beta1)) / delta
    y = (beta2 + beta1) * a / delta
    mu = brentq(lambda mu: mu**4 + 2*mu**3 + (1-x**2-y**2)*mu**2 - 2*y**2*mu - y**2, 1e-3, pi*a)
    alpha1 = atan2(-x / (1+mu), y/mu)
    return alpha1

def solve_vicenty(a, f, lambda12, phi1, phi2):
    """ Used to provide an initial guess in the case of nearly antipodal points.

    Parameters
    ----------
    a:          equatorial radius
    f:          flattening
    lambda12:   difference in longitudes (radians)
    phi1:       first latitude (radians)
    phi2:       second latitude (radians)

    Returns
    -------
    alpha1:     forward azimuth at first point
    alpha2:     forward azimuth at second point
    s12:        distance between points

    see Karney (2013) J. Geod. for details
    """
    eccn2 = f*(2-f)
    beta1 = atan((1-f) * tan(phi1))
    beta2 = atan((1-f) * tan(phi2))
    w = sqrt(1 - eccn2 * (0.5 * (cos(beta1) + cos(beta2)))**2)
    omega12 = lambda12 / w

    z1_r = cos(beta1)*sin(beta2) - sin(beta1)*cos(beta2)*cos(omega12)
    z1_i = cos(beta2)*sin(omega12)
    z1 = sqrt(z1_r**2 + z1_i**2)
    sigma12 = atan2(z1, sin(beta1)*sin(beta2) + cos(beta1)*cos(beta2)*cos(omega12))
    z2_r = -sin(beta1)*cos(beta2) + cos(beta1)*sin(beta2)*cos(omega12)
    z2_i = cos(beta1)*sin(omega12)

    alpha1 = atan2(z1_i, z1_r)
    alpha2 = atan2(z2_i, z2_r)
    s12 = a*w*sigma12
    return alpha1, alpha2, s12

def _degrees(r):
    return r*180/pi

def printd(*args):
    argsd = [_degrees(a) for a in args]
    print(*argsd)

LonLatWGS84 = LonLat(6378137.0, 6356752.314245)

class Point(object):
    _geotype = "Point"

    def __init__(self, vertex, crs=LonLatWGS84):
        self.vertex = vertex
        self.crs = crs

    @property
    def x(self):
        return self.vertex[0]

    @property
    def y(self):
        return self.vertex[1]

    def distance(self, other):
        _, _, distance = self.crs.inverse(self.x, self.y, other.x, other.y)
        return distance

class MultipointBase(object):

    def __init__(self, vertices, crs=LonLatWGS84):
        if getattr(vertices[0], "_geotype", None) == "Point":
            self.vertices = [pt.vertex for pt in vertices]
        else:
            self.vertices = vertices
        self.crs = crs

    def __len__(self):
        return len(self.vertices)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return type(self)(self.vertices[key], crs=self.crs)
        else:
            return Point(self.vertices[key], crs=self.crs)

    def __iter__(self):
        return (self[i] for i in range(len(self)))

    @property
    def coordinates(self):
        return list(*zip(*self.vertices))

class Multipoint(MultipointBase):
    _geotype = "Multipoint"

class Line(MultipointBase):
    _geotype = "Line"

if __name__ == "__main__":
    # Karney's Table 2 example for the forward problem
    print("Forward problem")
    x1, y1, baz = LonLatWGS84.forward(0.0, 40.0, 30.0, 10e6)
    print("solution:", 137.84490004377, 41.79331020506, 149.09016931807)
    print("computed:", x1, y1, baz-180)

    # vicenty problem, Table 3
    print("\nVicenty")
    a = 6378137.0
    f = 1/298.257223563
    phi1 = -30.12345*pi/180
    phi2 = -30.12344*pi/180
    lambda12 = 0.00005*pi/180
    alpha1, alpha2, distance = solve_vicenty(a, f, lambda12, phi1, phi2)
    print("solution", 77.043533, 77.043508, 4.944208)
    print("computed", alpha1*180/pi, alpha2*180/pi, distance)

    # astroid problem, Table 4
    print("\nAstroid")
    a = 6378137.0
    f = 1/298.257223563
    phi1 = -30*pi/180
    phi2 = 29.9*pi/180
    lambda12 = 179.8*pi/180
    alpha1 = solve_astroid(a, f, lambda12, phi1, phi2)
    print("solution:", 161.914)
    print("computed:", alpha1*180/pi)

    # full inverse problem, Table 5
    print("\nInverse problem")
    phi1 = -30
    phi2 = 29.9
    lambda12 = 179.8

    az, backaz, dist = LonLatWGS84.inverse(0.0, phi1, lambda12, phi2)
    print("solution:", 161.890524, 19989832.827610)
    print("computed:", az, dist)

    # full inverse problem with meridional points
    print("\nstress test")
    az, baz, d = LonLatWGS84.inverse(80.0, 8.0, -100.0, 8.0)
    print("solution:", 0.0, 0.0)
    print("computed:", az, baz)
