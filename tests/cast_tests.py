import unittest
import os
import numpy as np
import narwhal
from narwhal import gsw
from narwhal.cast import Cast, CTDCast, XBTCast, LADCP
from narwhal.cast import CastCollection
from narwhal.bathymetry import Bathymetry
from narwhal.util import force_monotonic, diff2, uintegrate, diff2_inner
from narwhal import util

directory = os.path.dirname(__file__)
DATADIR = os.path.join(directory, "data")
if not os.path.exists(DATADIR):
    os.mkdir(DATADIR)

class CastTests(unittest.TestCase):

    def setUp(self):
        p = np.arange(1, 1001, 2)
        temp = 10. * np.exp(-.008*p) - 15. * np.exp(-0.005*(p+100)) + 2.
        sal = -14. * np.exp(-.01*p) + 34.
        self.p = p
        self.temp = temp
        self.sal = sal
        self.cast1 = Cast(self.p, temp=self.temp, sal=self.sal)
        return

    def test_numerical_indexing(self):
        def _has_all(tup1, tup2):
            """ Check whether tup2 contains all items of tup1 """
            for item in tup1:
                if item not in tup2:
                    return False
            return True
        self.assertTrue(_has_all(self.cast1[40], (("pres", 81),
            ("temp", 1.1627808544797258), ("sal", 27.771987072878822))))
        self.assertTrue(_has_all(self.cast1[100], (("pres", 201),
            ("temp", 0.67261848597249019), ("sal", 32.124158554636729))))
        self.assertTrue(_has_all(self.cast1[400], (("pres", 801),
            ("temp", 1.8506793256302907), ("sal", 33.995350253934227))))
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

    def test_interpolate(self):
        self.assertEqual(np.round(self.cast1.interpolate("temp", "pres", 4.0), 8),
                         2.76745605)
        self.assertEqual(np.round(self.cast1.interpolate("temp", "sal", 33.0), 8),
                         0.77935861)
        #self.assertEqual(np.round(self.cast1.interpolate("pres", "temp", 1.5), 8),
        #                 2.7674560521632685)
        return

    def test_add_density(self):
        p = np.arange(10)
        t = 20.0 * 0.2 * p
        s = 30.0 * 0.25 * p
        x = [-20.0 for _ in p]
        y = [50.0 for _ in p]
        sa = gsw.sa_from_sp(s, p, x, y)
        ct = gsw.ct_from_t(sa, t, p)
        rho = gsw.rho(sa, ct, p)

        cast = CTDCast(p, s, t, coords=(-20, 50))
        cast.add_density()
        self.assertTrue(np.allclose(rho, cast["rho"]))
        return

    def test_add_buoyancy_freq_squared(self):
        # This is a fairly lousy test, merely ensuring that an N^2 field was
        # calculated, and that it's not wildly different than the direct
        # calculation.
        p = np.arange(10)
        t = 20.0 * 0.2 * p
        s = 30.0 * 0.25 * p
        x = [-20.0 for _ in p]
        y = [50.0 for _ in p]
        sa = gsw.sa_from_sp(s, p, x, y)
        ct = gsw.ct_from_t(sa, t, p)
        rho = np.asarray(gsw.rho(sa, ct, p))

        drhodz = -np.r_[rho[1]-rho[0], rho[2:]-rho[:-2], rho[-1]-rho[-2]] / \
                  np.r_[p[1]-p[0], p[2:]-p[:-2], p[-1]-p[-2]]
        N2_direct = -9.81 / rho * drhodz
        #N2_direct = np.convolve(N2_direct, 0.2*np.ones(5), mode="valid")

        cast = CTDCast(p, s, t, coords=(-20, 50), rho=rho)
        cast.add_Nsquared(rhokey="rho")
        #print(np.mean(np.abs(cast["N2"] - N2_direct)))
        self.assertTrue(np.mean(np.abs(cast["N2"] - N2_direct)) < 0.0003)
        return

    def test_LADCP_shear(self):
        z = np.arange(0, 300)
        u = z**1.01 - z
        v = z**1.005 - z
        u_ans = 1.01 * z**0.01 - 1
        v_ans = 1.005 * z**0.005 - 1
        lad = narwhal.LADCP(z, u, v)
        lad.add_shear()
        self.assertTrue(np.max(abs(lad["dudz"][1:-1] - u_ans[1:-1])) < 0.005)
        self.assertTrue(np.max(abs(lad["dvdz"][1:-1] - v_ans[1:-1])) < 0.005)
        return

    #def test_projectother(self):
    #    pass

    #def test_calculate_sigma(self):
    #    pass

    #def test_calculate_theta(self):
    #    pass

class CastCollectionTests(unittest.TestCase):

    def setUp(self):
        p = np.linspace(1, 999, 500)
        casts = []
        for i in range(10):
            cast = Cast(p, temp=2*np.ones_like(p), sal=30*np.ones_like(p))
            casts.append(cast)
        self.cc = CastCollection(casts)
        return

    def test_iteration1(self):
        for cast in self.cc:
            self.assertTrue(isinstance(cast, Cast))
        return

    def test_iteration2(self):
        for i, cast in enumerate(self.cc):
            pass
        self.assertEqual(i, 9)
        return

    def test_len(self):
        self.assertEqual(len(self.cc), 10)
        return

    def test_slicing(self):
        subcc = self.cc[2:7]
        self.assertTrue(isinstance(subcc, CastCollection))
        return

    pass

class MiscTests(unittest.TestCase):

    def test_force_monotonic(self):
        s = np.array([1, 3, 5, 6, 7, 9, 13, 14, 15])
        sm = force_monotonic(s)
        self.assertTrue(np.all(sm == s))

        s = np.array([1, 3, 5, 4, 7, 9, 13, 11, 15])
        sm = force_monotonic(s)
        self.assertTrue(np.all(sm == np.array([1, 3, 5, 5+1e-16, 7,
                                               9, 13, 13+1e-16, 15])))
        return

    def test_diff2(self):
        x = np.atleast_2d(np.linspace(-1, 1, 100))
        A = x**2 - (x + x.T)**3
        ans = 2*x - 3*(x + x.T)**2      # true answer

        # add holes
        A[30:40,1] = np.nan
        A[15,35] = np.nan
        A[30:50,50:55] = np.nan
        A[60:65,60] = np.nan
        A[60:70,-2] = np.nan

        D = diff2(A, x.ravel())
        self.assertTrue(np.max(abs(ans[~np.isnan(D)] - D[~np.isnan(D)])) < 0.15)
        return

    def test_diff2_inner(self):
        x = np.atleast_2d(np.linspace(-1, 1, 100))
        A = x**2 - (x + x.T)**3
        xinner = np.atleast_2d(0.5 * (x[0,1:] + x[0,:-1]))
        ans = 2*xinner - 3*(xinner + x.T)**2      # true answer

        # add holes
        A[30:40,1] = np.nan
        A[15,35] = np.nan
        A[30:50,50:55] = np.nan
        A[60:65,60] = np.nan
        A[60:70,-2] = np.nan

        D = diff2_inner(A, x.ravel())
        self.assertTrue(np.max(abs(ans[~np.isnan(D)] - D[~np.isnan(D)])) < 0.0002)
        return

    def test_diff2_dinterp(self):
        x = np.atleast_2d(np.linspace(-1, 1, 100))
        A = x**2 - (x + x.T)**3
        ans = 2*x - 3*(x + x.T)**2      # true answer

        # add holes
        A[30:40,1] = np.nan
        A[15,35] = np.nan
        A[30:50,50:55] = np.nan
        A[60:65,60] = np.nan
        A[60:70,-2] = np.nan

        D = util.diff2_dinterp(A, x.ravel())
        self.assertTrue(np.max(abs(ans[~np.isnan(D)] - D[~np.isnan(D)])) < 2.0)
        # Scheme has lousy accuracy in this test, but I think it's a fairly bad case
        return

    def test_uintegrate(self):
        x = np.atleast_2d(np.linspace(-1, 1, 100))
        ans = x**2 - (x + x.T)**3
        D = - 3*(x + x.T)**2

        I = uintegrate(D, np.tile(x.T, (1, x.size)), ubase=ans[-1])
        self.assertTrue(np.max(abs(ans - I)) < 0.0005)
        return

