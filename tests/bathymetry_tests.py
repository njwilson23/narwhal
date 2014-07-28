import unittest
from narwhal import Bathymetry, Cast, CastCollection
import karta
import numpy as np

try:
    LONLAT_WGS84 = karta.crsreg.LONLAT_WGS84
except AttributeError:
    LONLAT_WGS84 = karta.LONLAT_WGS84

class BathymetryTests(unittest.TestCase):

    def setUp(self):
        x = [-17.41933333, -17.42628333, -17.42573333, -17.4254,
             -17.42581667, -17.42583333, -17.4269    , -17.4437,
             -17.44126667, -17.44416667, -17.44673333, -17.46633333, -17.48418333]
        y = [80.07101667,  80.0878    ,  80.09245   ,  80.10168333,
             80.10895   ,  80.11108333,  80.11398333,  80.12305   ,
             80.12928333,  80.1431    ,  80.1534    ,  80.16636667,  80.16741667]
        depth = [102,  95,  90, 100, 110, 120, 130, 140, 150, 170, 160, 140, 130]
        self.bathymetry = Bathymetry(zip(x, y), depth=depth)
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
        correctresult = np.array([92.61373822294766, 123.15924954803165,
                                  150.24992416068667])
        depths = [c.properties["depth"] for c in cc]
        self.assertTrue(np.allclose(depths, correctresult))
        return

    def test_project_along_cruise(self):
        cruiseline = karta.Line([(0,0), (4,3), (6,2), (6,5)], crs=LONLAT_WGS84)
        bath = Bathymetry([(0,0), (2,1), (3,3), (5,3), (7,3), (5,4), (7,4.5)],
                          depth=[100, 120, 130, 135, 115, 127, 119])

        P, Q = bath.project_along_cruise(cruiseline)

        Pans = [0.,244562.46558282,465910.74224686,654663.50946077,
                914121.2637559,1024716.52731376,1080015.10269574]
        Qans = [0.,44497.10923469,66363.58647208,49450.41134166,
                111167.93504577,111050.1033939,110978.58197409]
        for (pa,qa),(p,q) in zip(zip(Pans, Qans), zip(P, Q)):
            self.assertAlmostEqual(pa, p, 4)
            self.assertAlmostEqual(qa, q, 4)
        return

