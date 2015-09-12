import unittest
import matplotlib
matplotlib.use("Agg")

from cast_tests import *
from analysis_tests import *
from geo_tests import *
from bathymetry_tests import *
from io_tests import *
from misc_tests import *
from plot_tests import *

if __name__ == "__main__":
    unittest.main()
