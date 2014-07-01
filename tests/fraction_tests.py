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

    def test_three_sources1(self):
        c = CTDCast(np.arange(10), 3.6*np.ones(10), 22.1*np.ones(10))
        r1, r2, r3 = [1, 5, 8], [25, 20, 18], [1, 1, 1]
        r1, r2, r3 = [1, 25], [5, 20], [8, 18]
        (f1, f2, f3) = c.water_fractions((r1, r2, r3))
        self.assertTrue(np.allclose(f1, 0.5*np.ones(10)))
        self.assertTrue(np.allclose(f2, 0.3*np.ones(10)))
        self.assertTrue(np.allclose(f3, 0.2*np.ones(10)))
        return

    def test_three_sources2(self):
        print("test incomplete")
        source1 = (10.0, 34.0)
        source2 = (2.0, 32.0)
        source3 = (17.0, 34.5)

        p = np.arange(0, 2000, 2)
        S = 34.3 - 2.0 * np.exp(-p/300.0)
        T = 15.0 * np.exp(-p/150.0) - 2e-3 * p
        cast = CTDCast(p, S, T)
        partitions = cast.water_fractions([source1, source2, source3])

        # import matplotlib.pyplot as plt
        # import seaborn
        # for partition in partitions:
        #     plt.plot(partition, p)
        # plt.show()
        
        # temporary
        self.assertTrue(partitions is not None)
        return

if __name__ == "__main__":
    unittest.main()

