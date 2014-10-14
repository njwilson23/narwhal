import narwhal
from narwhal import AbstractCast, AbstractCastCollection, CastCollection
from narwhal.plotting.plotutil import ensureiterable, count_casts
import numpy as np
import unittest


class PlotTests(unittest.TestCase):

    def setUp(self):

        # generate a cast collection
        cc = []
        p = np.arange(500)
        i = 0
        while i != 10:
            t = 15 * np.exp(-p/500) + 0.2 * abs(i-5)
            s = 4 * np.exp(-p/500) + 30 + 0.05 * abs(i-5)
            d = 400 - 2*(i-3)**2
            cast = narwhal.CTDCast(p, t, s, properties={"depth": d})
            i += 1
        self.cc = narwhal.CastCollection(cc)

        return

    def test_ensureiterable1(self):

        it = ensureiterable(["one", "two", "three"])
        ans = ["one", "two", "three", "one", "two", "three"]

        i = 0
        while i != 6:
            self.assertEqual(next(it), ans[i])
            i += 1
        return

    def test_ensureiterable2(self):

        it = ensureiterable(42)

        i = 0
        while i != 5:
            self.assertEqual(next(it), 42)
            i += 1
        return

    def test_count_casts(self):
        castlikes = [AbstractCast(),
                     AbstractCast(),
                     narwhal.CastCollection([AbstractCast(), AbstractCast()]),
                     AbstractCast()]
        self.assertEqual(count_casts(castlikes), 5)
        return

if __name__ == "__main__":
    unittest.main()
