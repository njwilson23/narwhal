import copy
import numbers
import numpy as np
import scipy.integrate as scint 
from narwhal.cast import Cast

def _nanmean(arr, axis=0):
    """ Re-implement nanmean in a way that doesn't fire off warning when there
    are NaN-filled rows. 
    
    Note that here axis is the axis to retain, which is not the behaviour of
    np.nanmean. I did it this way for simplicity, but this might be confusing
    in the future.
    """
    if axis != 0:
        arr = np.rollaxis(arr, axis)
    means = np.empty(arr.shape[0], dtype=arr.dtype)
    i = 0
    for row in arr:
        valid = row[~np.isnan(row)]
        if len(valid) > 0:
            means[i] = np.mean(row[~np.isnan(row)])
        else:
            means[i] = np.nan
        i += 1
    return means

def ccmeans(cc):
    """ Calculate a mean Cast along isopycnals from a CastCollection. """
    c0 = max(cc, key=lambda c: c.nvalid())
    s0 = c0["sigma"]
    sharedkeys = set(c0.data.keys()).intersection(
                    *[set(c.data.keys()) for c in cc[1:]]).difference(
                    set(("pres", "botdepth", "time")))
    nanmask = reduce(lambda a,b: a*b, [c.nanmask() for c in cc])
    data = dict()
    for key in sharedkeys:
        arr = np.nan * np.empty((len(cc), len(c0["pres"])))
        arr[0,:] = c0[key]
        for j, c in enumerate(cc[1:]):
            s = np.convolve(c["sigma"], np.ones(3)/3.0, mode="same")
            arr[j+1,:] = np.interp(s0, s, c[key])
        data[key] = _nanmean(arr, axis=1)
        data[key][nanmask] = np.nan

    return Cast(copy.copy(c0["pres"]), **data)

def ccmeanp(cc):
    if False in (np.all(cc[0]["pres"] == c["pres"]) for c in cc):
        raise ValueError("casts must share pressure levels")
    p = cc[0]["pres"]
    # shared keys are those in all casts, minus pressure and botdepth
    sharedkeys = set(cc[0].data.keys()).intersection(
                    *[set(c.data.keys()) for c in cc[1:]]).difference(
                    set(("pres", "botdepth", "time")))
    data = dict()
    for key in sharedkeys:
        arr = np.vstack([c.data[key] for c in cc])
        data[key] = _nanmean(arr, axis=1)

    return Cast(p, **data)

def force_monotonic(u):
    """ Given a nearly monotonically-increasing vector u, return a vector u'
    that is monotonic by incrementing each value u_i that is less than u_(i-1).

    u::iterable         vector to adjust
    """
    # naive implementation
    #v = u.copy()
    #for i in xrange(1, len(v)):
    #    if v[i] <= v[i-1]:
    #        v[i] = v[i-1] + 1e-16
    #return v

    # more efficient implementation
    v = [u1 if u1 > u0 else u0 + 1e-16
            for u0, u1 in zip(u[:-1], u[1:])]
    return np.hstack([u[0], v])

def diff1(V, x):
    """ Compute hybrid centred/sided difference of vector V with positions given by x """
    D = np.empty_like(V)
    D[1:-1] = (V[2:] - V[:-2]) / (x[2:] - x[:-2])
    D[0] = (V[1] - V[0]) / (x[1] - x[0])
    D[-1] = (V[-1] - V[-2]) / (x[-1] - x[-2])
    return D

def diff2(A, x):
    """ Return the row-wise differences in array A. Uses centred differences in
    the interior and one-sided differences on the edges. When there are
    interior NaNs, one-sided differences are used to fill in an much data as
    possible. """
    D2 = np.nan * np.empty_like(A)
    for (i, arow) in enumerate(A):
        start = -1
        for j in range(len(arow)):

            if start == -1 and ~np.isnan(arow[j]):
                start = j

            elif start != -1 and np.isnan(arow[j]):
                if j - start != 1:
                    D2[i,start:j] = diff1(arow[start:j], x[start:j])
                start = -1

            elif start != -1 and j == len(arow) - 1:
                D2[i,start:] = diff1(arow[start:], x[start:])
    return D2

def diff1_inner(V, x):
    """ Compute centred differences between points given by x """
    D = np.empty_like(len(V)-1)
    D = (V[1:] - V[:-1]) / (x[1:] - x[:-1])
    return D

def diff2_inner(A, x):
    """ Return the row-wise differences in array A. Uses centred differences in
    the interior and one-sided differences on the edges. When there are
    interior NaNs, one-sided differences are used to fill in an much data as
    possible. """
    (m, n) = A.shape
    D2 = np.nan * np.empty((m, n-1))
    for (i, arow) in enumerate(A):
        start = -1
        for j in range(len(arow)-1):

            if start == -1 and ~np.isnan(arow[j]):
                start = j

            elif start != -1 and np.isnan(arow[j]):
                if j - start != 1:
                    D2[i,start:j-1] = diff1_inner(arow[start:j], x[start:j])
                start = -1

            elif start != -1 and j == len(arow) - 2:
                D2[i,start:] = diff1_inner(arow[start:], x[start:])
    return D2

def uintegrate(dudz, X, ubase=0.0):
    """ Integrate velocity shear from the first non-NaN value to the top. """
    U = -np.nan*np.empty_like(dudz)
    if isinstance(ubase, numbers.Number):
        ubase = ubase * np.ones(dudz.shape[1], dtype=np.float64)
    
    for jcol in range(dudz.shape[1]):
        # find the deepest non-NaN
        imax = np.max(np.arange(dudz.shape[0])[~np.isnan(dudz[:,jcol])])
        du = dudz[:imax+1,jcol]
        du[np.isnan(du)] = 0.0
        U[:imax+1,jcol] = scint.cumtrapz(du, x=X[:imax+1,jcol],
                                         initial=0.0)
        U[:imax+1,jcol] -= U[imax,jcol] - ubase[jcol]
    return U
