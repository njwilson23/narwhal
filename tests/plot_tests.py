import os.path
import narwhal
from narwhal import AbstractCast, AbstractCastCollection, CastCollection
from narwhal.plotting.plotutil import ensureiterable, count_casts
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import unittest

matplotlib.use("Agg")       # headless

directory = os.path.dirname(__file__)
DATADIR = os.path.join(directory, "data")

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
                     CastCollection([AbstractCast(), AbstractCast()]),
                     AbstractCast()]
        self.assertEqual(count_casts(castlikes), 5)
        return

class SectionPlotTests(unittest.TestCase):
    def setUp(self):

        self.ctds = narwhal.load_json(os.path.join(DATADIR, "line_w_dec1997.nwl"))

    def test_label_station_strings(self):
        ax = plt.axes(projection="section")
        txts = ax.label_stations(self.ctds,
                                 [str(i+1) for i in range(len(self.ctds))])
        for i, txt in enumerate(txts):
            self.assertEqual(str(i+1), txt.get_text())
        return

    def test_label_station_function(self):
        ax = plt.axes(projection="section")
        def flabel(cast):
            return "Station %i" % cast.properties["id"]
        txts = ax.label_stations(self.ctds,flabel)

        stations = [9005, 9006, 9007, 9009, 9010, 9011]
        for i, txt in enumerate(txts):
            self.assertEqual("Station " + str(stations[i]), txt.get_text())
        return

    def test_label_station_key(self):
        ax = plt.axes(projection="section")
        txts = ax.label_stations(self.ctds, "id")

        stations = [9005, 9006, 9007, 9009, 9010, 9011]
        for i, txt in enumerate(txts):
            self.assertEqual(str(stations[i]), txt.get_text())
        return

if __name__ == "__main__":
    unittest.main()
