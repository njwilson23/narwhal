import numbers
import numpy as np
import scipy.integrate as scint 

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
