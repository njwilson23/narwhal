""" Generate reference data for automated tests """

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
        self.cast = Cast(self.p, temp=self.temp, sal=self.sal, date=dt)
        self.ctd = CTDCast(self.p, temp=self.temp, sal=self.sal, date=dt)
        self.ctdz = CTDCast(self.p, temp=self.temp, sal=self.sal, date=dt)
        self.xbt = XBTCast(self.p, temp=self.temp, sal=self.sal, date=dt)
        self.collection = CastCollection(self.ctd, self.xbt, self.ctd)
        return

    def save_nwl(self):
        self.cast.save("data/reference_cast_test.nwl", binary=False)
        self.ctd.save("data/reference_ctd_test.nwl", binary=False)
        self.ctdz.save("data/reference_ctdz_test.nwl", binary=False)
        self.xbt.save("data/reference_xbt_test.nwl", binary=False)
        self.collection.save("data/reference_coll_test.nwl", binary=False)

    def save_nwz(self):
        self.cast.save("data/reference_cast_test.nwz")
        self.ctd.save("data/reference_ctd_test.nwz")
        self.ctdz.save("data/reference_ctdz_test.nwz")
        self.xbt.save("data/reference_xbt_test.nwz")
        self.collection.save("data/reference_coll_test.nwz")

if __name__ == "__main__":
    data = TestData()
    data.save_nwl()
    data.save_nwz()

