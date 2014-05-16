
import unittest
import numpy as np
from narwhal.cast import Cast, CastCollection
from narwhal.cast import force_monotonic
from narwhal.bathymetry import Bathymetry

class CastTests(unittest.TestCase):

    def setUp(self):
        p = np.arange(1, 1001, 2)
        temp = 10. * np.exp(-.008*p) - 15. * np.exp(-0.005*(p+100)) + 2.
        sal = -14. * np.exp(-.01*p) + 34.
        self.p = p
        self.temp = temp
        self.sal = sal
        self.cast1 = Cast(self.p, temp=self.temp, sal=self.sal)
        return

    def test_numerical_indexing(self):
        def _has_all(tup1, tup2):
            """ Check whether tup2 contains all items of tup1 """
            for item in tup1:
                if item not in tup2:
                    return False
            return True
        self.assertTrue(_has_all(self.cast1[40], (("pres", 81),
            ("temp", 1.1627808544797258), ("sal", 27.771987072878822))))
        self.assertTrue(_has_all(self.cast1[100], (("pres", 201),
            ("temp", 0.67261848597249019), ("sal", 32.124158554636729))))
        self.assertTrue(_has_all(self.cast1[400], (("pres", 801),
            ("temp", 1.8506793256302907), ("sal", 33.995350253934227))))
        return

    def test_kw_indexing(self):
        self.assertTrue(np.all(self.cast1["pres"] == self.p))
        self.assertTrue(np.all(self.cast1["sal"] == self.sal))
        self.assertTrue(np.all(self.cast1["temp"] == self.temp))
        return

    def test_concatenation(self):
        p = np.arange(1, 1001, 2)
        temp = 12. * np.exp(-.007*p) - 14. * np.exp(-0.005*(p+100)) + 1.8
        sal = -13. * np.exp(-.01*p) + 34.5
        cast2 = Cast(p, temp=temp, sal=sal)
        cc = self.cast1 + cast2
        self.assertTrue(isinstance(cc, CastCollection))
        self.assertEqual(len(cc), 2)
        return

    def test_interpolate(self):
        self.assertEqual(np.round(self.cast1.interpolate("temp", "pres", 4.0), 8),
                         2.76745605)
        self.assertEqual(np.round(self.cast1.interpolate("temp", "sal", 33.0), 8),
                         0.77935861)
        #self.assertEqual(np.round(self.cast1.interpolate("pres", "temp", 1.5), 8),
        #                 2.7674560521632685)
        return

    def test_projectother(self):
        pass

    def test_calculate_sigma(self):
        pass

    def test_calculate_theta(self):
        pass

class CastCollectionTests(unittest.TestCase):

    def setUp(self):
        p = np.linspace(1, 999, 500)
        casts = []
        for i in range(10):
            cast = Cast(p, temp=2*np.ones_like(p), sal=30*np.ones_like(p))
            casts.append(cast)
        self.cc = CastCollection(casts)
        return

    def test_iteration1(self):
        for cast in self.cc:
            self.assertTrue(isinstance(cast, Cast))
        return

    def test_iteration2(self):
        for i, cast in enumerate(self.cc):
            pass
        self.assertEqual(i, 9)
        return

    def test_len(self):
        self.assertEqual(len(self.cc), 10)
        return

    def test_slicing(self):
        subcc = self.cc[2:7]
        self.assertTrue(isinstance(subcc, CastCollection))
        return

    pass

class BathymetryTests(unittest.TestCase):

    def setUp(self):
        x = [-17.41933333, -17.42628333, -17.42573333, -17.4254,
             -17.42581667, -17.42583333, -17.4269    , -17.4437,
             -17.44126667, -17.44416667, -17.44673333, -17.46633333, -17.48418333]
        y = [80.07101667,  80.0878    ,  80.09245   ,  80.10168333,
             80.10895   ,  80.11108333,  80.11398333,  80.12305   ,
             80.12928333,  80.1431    ,  80.1534    ,  80.16636667,  80.16741667]
        depth = [102,  95,  90, 100, 110, 120, 130, 140, 150, 170, 160, 140, 130]
        self.bathymetry = Bathymetry(x, y, depth)
        return

    def test_add_to_castcollection(self):
        cc = CastCollection(
                Cast(np.arange(100), T=np.random.rand(100), S=np.random.rand(100),
                     coords=(-17.42, 80.09)),
                Cast(np.arange(100), T=np.random.rand(100), S=np.random.rand(100),
                     coords=(-17.426, 80.112)),
                Cast(np.arange(100), T=np.random.rand(100), S=np.random.rand(100),
                     coords=(-17.45, 80.16)))
        cc.add_bathymetry(self.bathymetry)
        correctresult = np.array([91.87894384, 122.9704543, 154.58219622])
        self.assertTrue(np.allclose(cc["botdepth"], correctresult))
        return

class MiscTests(unittest.TestCase):

    def test_force_monotonic(self):
        s = np.array([1, 3, 5, 6, 7, 9, 13, 14, 15])
        sm = force_monotonic(s)
        self.assertTrue(np.all(sm == s))

        s = np.array([1, 3, 5, 4, 7, 9, 13, 11, 15])
        sm = force_monotonic(s)
        self.assertTrue(np.all(sm == np.array([1, 3, 5, 5+1e-16, 7,
                                               9, 13, 13+1e-16, 15])))
        return

if __name__ == "__main__":
    unittest.main()
