import unittest
from math import pi
import random
import narwhal.geo
from narwhal.geo import Point, Multipoint, Line, LonLatWGS84

class TestGeometry(unittest.TestCase):

    def test_create_point(self):
        # tests that a Point can be instantiated
        point = Point((-70.672439, 41.524079), crs=LonLatWGS84)
        self.assertTrue(isinstance(point, Point))
        return

    def test_create_line(self):
        # tests that a Line can be instantiated, sliced, and iterated through,
        # producing points
        line = Line([(70.205, 40.283166666666666),
                      (70.205, 40.283),
                      (70.205, 40.282666666666664),
                      (70.1035, 40.14033333333333),
                      (70.10316666666667, 40.14),
                      (70.10283333333334, 40.13966666666666),
                      (70.00566666666667, 40.0115),
                      (70.0055, 40.0115),
                      (70.00516666666667, 40.011),
                      (69.92983333333333, 39.899166666666666),
                      (69.9295, 39.899166666666666),
                      (69.92833333333333, 39.899166666666666),
                      (69.902, 39.8585),
                      (69.90116666666667, 39.858666666666664),
                      (69.89966666666666, 39.86),
                      (69.85133333333333, 39.791333333333334),
                      (69.85133333333333, 39.791333333333334),
                      (69.85116666666667, 39.791),
                      (69.7995, 39.70066666666666),
                      (69.79933333333334, 39.70033333333333),
                      (69.80066666666667, 39.6955),
                      (69.6525, 39.474333333333334),
                      (69.65333333333334, 39.47416666666667),
                      (69.66433333333333, 39.47),
                      (69.48716666666667, 39.266333333333336),
                      (69.4875, 39.26683333333333),
                      (69.48966666666666, 39.2675),
                      (69.33383333333333, 39.0145),
                      (69.33416666666666, 39.01416666666667),
                      (69.33933333333333, 39.01083333333333),
                      (69.18516666666666, 38.790333333333336),
                      (69.18516666666666, 38.790333333333336),
                      (69.18316666666666, 38.79),
                      (69.0255, 38.5595),
                      (69.02533333333334, 38.5595),
                      (69.02366666666667, 38.55866666666667),
                      (68.861, 38.33116666666667),
                      (68.861, 38.3315),
                      (68.8585, 38.33683333333333),
                      (68.6975, 38.099666666666664),
                      (68.69716666666666, 38.10066666666667),
                      (68.69033333333333, 38.13183333333333),
                      (68.54116666666667, 37.85066666666667),
                      (68.54116666666667, 37.850833333333334),
                      (68.55366666666667, 37.867333333333335),
                      (68.38033333333334, 37.62233333333333),
                      (68.381, 37.62233333333333),
                      (68.41916666666667, 37.64083333333333),
                      (68.22916666666667, 37.385666666666665),
                      (68.2305, 37.38616666666667),
                      (68.27933333333333, 37.4055),
                      (68.06283333333333, 37.14333333333333),
                      (68.06383333333333, 37.14516666666667),
                      (68.09783333333333, 37.18383333333333),
                      (67.899, 36.902),
                      (67.89916666666667, 36.903333333333336),
                      (67.92, 36.945166666666665),
                      (67.7355, 36.66116666666667),
                      (67.7345, 36.663),
                      (67.71333333333334, 36.69566666666667),
                      (67.45166666666667, 36.19916666666666),
                      (67.45183333333334, 36.199),
                      (67.45566666666667, 36.21333333333333),
                      (67.161, 35.710166666666666),
                      (67.16083333333333, 35.709666666666664),
                      (67.1625, 35.709666666666664),
                      (66.86983333333333, 35.227333333333334),
                      (66.87, 35.227333333333334),
                      (66.88, 35.23283333333333),
                      (66.57966666666667, 34.74),
                      (66.57983333333334, 34.74),
                      (66.59233333333333, 34.739333333333335),
                      (66.28983333333333, 34.25983333333333),
                      (66.28983333333333, 34.25983333333333),
                      (66.2895, 34.25983333333333),
                      (65.99933333333334, 33.7785),
                      (65.99983333333333, 33.778166666666664),
                      (66.005, 33.781),
                      (65.69233333333334, 33.068),
                      (65.69233333333334, 33.068),
                      (65.68883333333333, 33.067),
                      (65.33266666666667, 32.58316666666666),
                      (65.33266666666667, 32.58316666666666),
                      (65.32816666666666, 32.58016666666666),
                      (65.22433333333333, 32.1625),
                      (65.22416666666666, 32.1625),
                      (65.22166666666666, 32.16683333333334)], crs=LonLatWGS84)
        self.assertTrue(isinstance(line, Line))
        self.assertEqual(len(line), 87)
        for pt in line[5:12]:
            self.assertTrue(isinstance(pt, Point))
            self.assertEqual(pt.crs, line.crs)
        return

    def test_distance(self):
        pt = Point((-140.0, 41.0))
        pt_northwest = Point((-142.0, 42.0))
        d = pt.distance(pt_northwest)
        self.assertLess(abs(d - 200544.120615), 0.1)    # from geographiclib
        return

    def test_distance_meridional(self):
        pt = Point((-140.0, 41.0))
        pt_north = Point((-140.0, 41.5))
        d = pt.distance(pt_north)
        self.assertLess(abs(d - 55529.372145), 0.1)     # from geographicslib
        return

    def test_distance_equatorial(self):
        pt = Point((-140.0, 0.0))
        pt_west = Point((-144.0, 0.0))
        d = pt.distance(pt_west)
        self.assertLess(abs(d - 445277.963), 1.0)       # from geod 4.8.0
        return


class TestGeodesy(unittest.TestCase):

    def test_normalize_longitude(self):
        self.assertEqual(narwhal.geo._normalize_longitude(45), 45)
        self.assertEqual(narwhal.geo._normalize_longitude(-45), -45)
        self.assertEqual(narwhal.geo._normalize_longitude(190), -170)
        self.assertEqual(narwhal.geo._normalize_longitude(470), 110)
        self.assertEqual(narwhal.geo._normalize_longitude(180), -180)
        self.assertAlmostEqual(narwhal.geo._normalize_longitude(179.9999), 179.9999)
        return

    def test_canonical_configuration(self):
        random.seed(42)
        for i in range(10):
            x1 = random.randint(-180, +179)
            x2 = random.randint(-180, +179)
            y1 = random.randint(-90, +90)
            y2 = random.randint(-90, +90)

            tr, x1t, y1t, x2t, y2t = narwhal.geo._canonical_configuration(x1, y1, x2, y2)
            self.assertTrue(y1t <= 0)
            self.assertTrue(y1t <= y2t <= -y1t)
            self.assertTrue(0 <= x2t-x1t <= 180)

    def test_forward(self):
        # solution from Karney table 2
        x1, y1, baz = LonLatWGS84.forward(0.0, 40.0, 30.0, 10e6)
        self.assertAlmostEqual(x1, 137.84490004377)
        self.assertAlmostEqual(y1, 41.79331020506)
        self.assertAlmostEqual(baz-180, 149.09016931807)
        return

    def test_vicenty(self):
        # soluton from Karney Table 3
        a = 6378137.0
        f = 1/298.257223563
        phi1 = -30.12345*pi/180
        phi2 = -30.12344*pi/180
        lambda12 = 0.00005*pi/180
        alpha1, alpha2, distance = narwhal.geo.solve_vicenty(a, f, lambda12, phi1, phi2)
        self.assertAlmostEqual(alpha1, 77.043533*pi/180.0)
        self.assertAlmostEqual(alpha2, 77.043508*pi/180.0)
        self.assertAlmostEqual(distance, 4.944208, places=6)

    def test_astroid(self):
        # solution from Karney Table 4
        a = 6378137.0
        f = 1/298.257223563
        phi1 = -30*pi/180
        phi2 = 29.9*pi/180
        lambda12 = 179.8*pi/180
        alpha1 = narwhal.geo.solve_astroid(a, f, lambda12, phi1, phi2)
        self.assertAlmostEqual(alpha1, 161.914*pi/180.0, places=4)

    def test_inverse(self):
        phi1 = -30
        phi2 = 29.9
        lambda12 = 179.8

        az, backaz, dist = LonLatWGS84.inverse(0.0, phi1, lambda12, phi2)
        self.assertAlmostEqual(az, 161.890524, places=5)
        self.assertAlmostEqual(dist, 19989832.827610, places=3)  # accurate to mm
        return

    def test_forward_azimuths(self):
        random.seed(84)

        baz_ans = [92.91388243367805,
                   -4.201720861613779,
                   52.80485809364322,
                   -142.7524103634052,
                   141.85941358467724,
                   -24.07561143948638,
                   -160.49527235357542,
                   -34.199026159594894,
                   -175.03051252479207,
                   -177.03930317086468,
                   -152.6533676960886,
                   -116.65718663523]

        for i in range(12):
            x1 = random.randint(-180, 179)
            y1 = random.randint(-90, 90)
            az = random.randint(0, 359)
            dist = random.randint(1e5, 5e7)

            _, _, baz = LonLatWGS84.forward(x1, y1, az, dist)
            self.assertAlmostEqual(narwhal.geo._normalize_longitude(baz), baz_ans[i])
            return

    def test_inverse_azimuths(self):
        random.seed(84)

        az_ans = [-177.96558522172995,
                  128.55935150957538,
                  153.79051165122308,
                  78.28417585048585,
                  -119.08597433579152,
                  6.878393066563715,
                  167.83620023616692,
                  -60.883596473778525,
                  146.80260470446478,
                  90.72563989100766,
                  91.52463811247357,
                  -176.57683616878396]

        baz_ans = [124.80226823425508,
                   -84.35400639349773,
                   -15.851049504289108,
                   -43.21579038193292,
                   58.745278161362435,
                   -22.04237984144868,
                   -131.24922341012208,
                   82.12804031596862,
                   -128.75928938927842,
                   -123.47642579406025,
                   -148.877863894165,
                   6.763244985451507]

        normalize = lambda x: (x+180) % 360 - 180

        for i in range(12):
            x1 = random.randint(-180, 179)
            x2 = random.randint(-180, 179)
            y1 = random.randint(-89, 89)
            y2 = random.randint(-89, 89)

            az, baz, d = LonLatWGS84.inverse(x1, y1, x2, y2)
            self.assertAlmostEqual(normalize(az), normalize(az_ans[i]))
            self.assertAlmostEqual(normalize(baz), normalize(baz_ans[i]))

    def test_inverse_meridional_case2(self):
        # Test a few difficult cases that arose during development
        az, baz, _ = LonLatWGS84.inverse(100, 8, -80, 8)
        self.assertAlmostEqual(az, 0.0)
        self.assertAlmostEqual(baz, 0.0)

        az, baz, _ = LonLatWGS84.inverse(-15, 86, 165, 86)
        self.assertAlmostEqual(az, 0.0)
        self.assertAlmostEqual(baz, 0.0)

        az, baz, _ = LonLatWGS84.inverse(-139, -23, 41, -23)
        self.assertAlmostEqual(az, 180.0)
        self.assertAlmostEqual(baz, 180.0)

if __name__ == "__main__":
    unittest.main()
