import unittest
import io
import os
import datetime
import numpy as np
import narwhal
from narwhal.cast import Cast, CTDCast, XBTCast, LADCP
from narwhal.cast import CastCollection

directory = os.path.dirname(__file__)
DATADIR = os.path.join(directory, "data")

class IOTests(unittest.TestCase):

    def setUp(self):
        p = np.arange(1, 1001, 2)
        temp = 10. * np.exp(-.008*p) - 15. * np.exp(-0.005*(p+100)) + 2.
        sal = -14. * np.exp(-.01*p) + 34.
        self.p = p
        self.temp = temp
        self.sal = sal
        dt = datetime.datetime(1993, 8, 18, 14, 42, 36)
        self.cast = Cast(self.p, temp=self.temp, sal=self.sal,
                        properties={"date":dt})
        self.ctd = CTDCast(self.p, temp=self.temp, sal=self.sal,
                        properties={"date":dt})
        self.xbt = XBTCast(self.p, temp=self.temp, sal=self.sal,
                        properties={"date":dt})
        self.collection = CastCollection(self.ctd, self.xbt, self.ctd)
        return

    def assertFilesEqual(self, f1, f2):
        f1.seek(0)
        f2.seek(0)
        s1 = f1.read()
        s2 = f2.read()
        self.assertEqual(s1, s2)
        return

    def test_save(self):
        #fnm = os.path.join(DATADIR, "cast_test.nwl")
        try:
            f = io.StringIO()
            self.cast.save(f)
        finally:
            f.close()

        #fnm = os.path.join(DATADIR, "ctd_test.nwl")
        try:
            f = io.StringIO()
            self.ctd.save(f)
        finally:
            f.close()

        #fnm = os.path.join(DATADIR, "xbt_test.nwl")
        try:
            f = io.StringIO()
            self.xbt.save(f)
        finally:
            f.close()
        return

    def test_save_collection(self):
        #fnm = os.path.join(DATADIR, "coll_test.nwl")
        try:
            f = io.StringIO()
            self.collection.save(f)
        finally:
            f.close()
        return

    def test_save_zprimarykey(self):
        cast = Cast(np.arange(len(self.p)), temp=self.temp, sal=self.sal,
                    primarykey="z", properties={})
        f = io.StringIO()
        try:
            cast.save(f)
        finally:
            f.close()
        return

    def test_load(self):
        cast = narwhal.read(os.path.join(DATADIR, "reference_cast_test.nwl"))
        ctd = narwhal.read(os.path.join(DATADIR, "reference_ctd_test.nwl"))
        xbt = narwhal.read(os.path.join(DATADIR, "reference_xbt_test.nwl"))
        self.assertEqual(cast, self.cast)
        self.assertEqual(ctd, self.ctd)
        self.assertEqual(xbt, self.xbt)
        return

    def test_load_collection(self):
        coll = narwhal.read(os.path.join(DATADIR, "reference_coll_test.nwl"))
        self.assertEqual(coll, self.collection)
        return

    def test_load_zprimarykey(self):
        castl = narwhal.read(os.path.join(DATADIR, "reference_ctdz_test.nwl"))
        cast = Cast(np.arange(len(self.p)), temp=self.temp, sal=self.sal,
                    primarykey="z", properties={})
        self.assertEqual(cast, castl)

