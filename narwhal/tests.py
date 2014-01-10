
import unittest
import numpy as np
from cast import Cast, CastCollection

class CastTests(unittest.TestCase):

    def setUp(self):
        p = np.arange(1, 1001, 2)
        temp = 10. * np.exp(-.008*p) - 15. * np.exp(-0.005*(p+100)) + 2.
        sal = -14. * np.exp(-.01*p) + 34.
        self.p = p
        self.temp = temp
        self.sal = sal
        self.cast1 = Cast(self.p, T=self.temp, S=self.sal)
        return

    def test_numerical_indexing(self):
        self.assertEqual(self.cast1[40],
                         (81, 27.771987072878822, 1.1627808544797258))
        self.assertEqual(self.cast1[100],
                         (201, 32.124158554636729, 0.67261848597249019))
        self.assertEqual(self.cast1[400],
                         (801, 33.995350253934227, 1.8506793256302907))
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

    def test_calculate_sigma(self):
        pass

    def test_calculate_theta(self):
        pass

class CastCollectionTests(unittest.TestCase):
    pass


if __name__ == "__main__":
    unittest.main()
