import unittest
import os
import numpy as np
import narwhal
from narwhal.cast import CTDCast

directory = os.path.dirname(__file__)
DATADIR = os.path.join(directory, "data")

class WaterFractionTests(unittest.TestCase):

    def setUp(self):
        pass

    def test_three_sources_constant(self):
        ans = np.array([0.3, 0.2, 0.5])
        sources = [(34.0, 2.0), (34.5, 7.0), (34.6, 5.0)]
        s = np.array(sources)
        x = np.ones(10, dtype=np.float64)
        sal = x * np.dot(ans, s[:,0])
        tmp = x * np.dot(ans, s[:,1])

        c = CTDCast(np.arange(10), sal, tmp)
        (chi1, chi2, chi3) = c.water_fractions(sources)
        self.assertTrue(np.allclose(chi1, 0.3*np.ones(10)))
        self.assertTrue(np.allclose(chi2, 0.2*np.ones(10)))
        self.assertTrue(np.allclose(chi3, 0.5*np.ones(10)))
        return

    def test_three_sources_varying(self):
        ans_chi1 = np.linspace(0.2, 0.35, 10)
        ans_chi2 = np.linspace(0.6, 0.1, 10)
        ans_chi3 = 1.0 - (ans_chi1 + ans_chi2)
        ans = np.c_[ans_chi1, ans_chi2, ans_chi3]

        sources = [(34.0, 2.0), (34.5, 7.0), (34.6, 5.0)]
        s = np.array(sources)
        x = np.ones(10, dtype=np.float64)
        sal = x * np.dot(ans, s[:,0])
        tmp = x * np.dot(ans, s[:,1])

        c = CTDCast(np.arange(10), sal, tmp)
        (chi1, chi2, chi3) = c.water_fractions(sources)
        self.assertTrue(np.allclose(chi1, ans_chi1))
        self.assertTrue(np.allclose(chi2, ans_chi2))
        self.assertTrue(np.allclose(chi3, ans_chi3))
        return

    def test_four_sources_varying(self):
        ans_chi1 = np.linspace(0.2, 0.35, 10)
        ans_chi2 = np.linspace(0.6, 0.1, 10)
        ans_chi3 = np.linspace(0.05, 0.12, 10)
        ans_chi4 = 1.0 - (ans_chi1 + ans_chi2 + ans_chi3)
        ans = np.c_[ans_chi1, ans_chi2, ans_chi3, ans_chi4]

        sources = [(34.0, 2.0, 280.0), (34.5, 70.0, 250.0),
                   (34.6, 5.0, 330.0), (33.9, 18.0, 390.0)]
        s = np.array(sources)
        x = np.ones(10, dtype=np.float64)
        sal = x * np.dot(ans, s[:,0])
        tmp = x * np.dot(ans, s[:,1])
        oxy = x * np.dot(ans, s[:,2])

        c = CTDCast(np.arange(10), sal, tmp, oxygen=oxy)
        (chi1, chi2, chi3, chi4) = c.water_fractions(sources,
                                    tracers=["salinity", "temperature", "oxygen"])
        self.assertTrue(np.allclose(chi1, ans_chi1))
        self.assertTrue(np.allclose(chi2, ans_chi2))
        self.assertTrue(np.allclose(chi3, ans_chi3))
        self.assertTrue(np.allclose(chi4, ans_chi4))
        return

if __name__ == "__main__":
    unittest.main()

