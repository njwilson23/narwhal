""" Generate reference data for automated tests """

from os.path import realpath, split, join
from narwhal import Cast, CTDCast, XBTCast, CastCollection
import datetime
import numpy as np

class TestData():

    def __init__(self):
        p = np.arange(1, 1001, 2)
        temp = 10. * np.exp(-.008*p) - 15. * np.exp(-0.005*(p+100)) + 2.
        sal = -14. * np.exp(-.01*p) + 34.
        self.p = p
        self.temp = temp
        self.sal = sal
        dt = datetime.datetime(1993, 8, 18, 14, 42, 36)
        self.cast = Cast(pres=self.p, temp=self.temp, sal=self.sal, date=dt)
        self.collection = CastCollection(self.cast, self.cast, self.cast)

        self.path = join(split(realpath(__file__))[0], "data")
        return

    def save_nwl(self):
        self.cast.save_json(join(self.path, "reference_cast_test.nwl"), binary=False)
        self.collection.save_json(join(self.path, "reference_coll_test.nwl"), binary=False)

    def save_nwz(self):
        self.cast.save_json(join(self.path, "reference_cast_test.nwz"))
        self.collection.save_json(join(self.path, "reference_coll_test.nwz"))

    def save_hdf(self):
        self.cast.save_hdf(join(self.path, "reference_cast_test.h5"))
        self.collection.save_hdf(join(self.path, "reference_coll_test.h5"))

if __name__ == "__main__":
    data = TestData()
    data.save_nwl()
    data.save_nwz()
    try:
        data.save_hdf()
    except IOError:
        print("HDF data not saved because h5py not installed")
