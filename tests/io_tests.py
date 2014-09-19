import unittest
import os
import sys
import datetime
import numpy as np
import narwhal
from narwhal.cast import Cast, CTDCast, XBTCast, LADCP
from narwhal.cast import CastCollection

from io import BytesIO
if sys.version_info[0] < 3:
    from cStringIO import StringIO
else:
    from io import StringIO

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
        self.cast = Cast(self.p, temp=self.temp, sal=self.sal, date=dt)
        self.ctd = CTDCast(self.p, temp=self.temp, sal=self.sal, date=dt)
        self.xbt = XBTCast(self.p, temp=self.temp, sal=self.sal, date=dt)
        self.collection = CastCollection(self.ctd, self.xbt, self.ctd)
        return

    def assertFilesEqual(self, f1, f2):
        f1.seek(0)
        f2.seek(0)
        s1 = f1.read()
        s2 = f2.read()
        self.assertEqual(s1, s2)
        return

    def test_save_text(self):
        try:
            f = StringIO()
            self.cast.save(f, binary=False)
        finally:
            f.close()

        try:
            f = StringIO()
            self.ctd.save(f, binary=False)
        finally:
            f.close()

        try:
            f = StringIO()
            self.xbt.save(f, binary=False)
        finally:
            f.close()
        return

    def test_save_binary(self):
        try:
            f = BytesIO()
            self.cast.save(f)
        finally:
            f.close()

        try:
            f = BytesIO()
            self.ctd.save(f)
        finally:
            f.close()

        try:
            f = BytesIO()
            self.xbt.save(f)
        finally:
            f.close()
        return

    def test_save_collection_text(self):
        try:
            f = StringIO()
            self.collection.save(f, binary=False)
        finally:
            f.close()
        return

    def test_save_collection_binary(self):
        try:
            f = BytesIO()
            self.collection.save(f)
        finally:
            f.close()
        return

    def test_save_zprimarykey(self):
        cast = Cast(np.arange(len(self.p)), temp=self.temp, sal=self.sal,
                    primarykey="z", properties={})
        f = BytesIO()
        try:
            cast.save(f)
        finally:
            f.close()
        return

    def test_load_text(self):
        cast = narwhal.read(os.path.join(DATADIR, "reference_cast_test.nwl"))
        ctd = narwhal.read(os.path.join(DATADIR, "reference_ctd_test.nwl"))
        xbt = narwhal.read(os.path.join(DATADIR, "reference_xbt_test.nwl"))
        self.assertEqual(cast, self.cast)
        self.assertEqual(ctd, self.ctd)
        self.assertEqual(xbt, self.xbt)
        return

    def test_load_binary(self):
        cast = narwhal.read(os.path.join(DATADIR, "reference_cast_test.nwz"))
        ctd = narwhal.read(os.path.join(DATADIR, "reference_ctd_test.nwz"))
        xbt = narwhal.read(os.path.join(DATADIR, "reference_xbt_test.nwz"))
        self.assertEqual(cast, self.cast)
        self.assertEqual(ctd, self.ctd)
        self.assertEqual(xbt, self.xbt)
        return

    def test_load_collection_text(self):
        coll = narwhal.read(os.path.join(DATADIR, "reference_coll_test.nwl"))
        self.assertEqual(coll, self.collection)
        return

    def test_load_collection_binary(self):
        coll = narwhal.read(os.path.join(DATADIR, "reference_coll_test.nwz"))
        self.assertEqual(coll, self.collection)
        return

#    def test_load_zprimarykey(self):
#        castl = narwhal.read(os.path.join(DATADIR, "reference_ctdz_test.nwl"))
#        cast = CTDCast(self.p, temp=self.temp, sal=self.sal,
#                       primarykey="z", properties={})
#        self.assertEqual(cast, castl)

