# -*- coding: utf-8 -*-
import numbers
import numpy as np
import scipy.integrate as scint
from scipy.ndimage.morphology import binary_dilation
from scipy import sparse

def sparse_diffmat(n, deriv, h, order=2):
    """ Return an `n::Int` by `n` sparse difference matrix to approximate a
    `deriv::int` derivative with spacing `h::float` to `order::int`-order
    accuracy. """
    if hasattr(h, "__len__"):
        raise NotImplementedError("only evenly spaced data are supported right now")
    if deriv == 1 and order == 2:
        I = np.ones(n)
        I_ = np.ones(n-1)
        D = sparse.diags((0.5*I_, -0.5*I_), (1, -1)) / h
        D = D.tolil()
        D[0,:3] = np.asarray([-1.5, 2, -0.5]) / h
        D[-1,-3:] = np.asarray([-0.5, 2, -1.5]) / h

    elif deriv == 2 and order == 2:
        I = np.ones(n)
        I_ = np.ones(n-1)
        D = sparse.diags((I_, -2*I, I_), (1, 0, -1)) / h**2
        D = D2.tolil()
        D[0,:4] = np.asarray([2, -5, 4, -1]) / h**2
        D[-1,-4:] = np.asarray([-1, 4, -5, 2]) / h**2
    else:
        raise NotImplementedError("{0} order {1}-derivative".format(order, deriv))
    return D

def force_monotonic(u):
    """ Given a nearly monotonically-increasing vector u, return a vector u'
    that is monotonic by incrementing each value u_i that is less than u_(i-1).

    u::iterable         vector to adjust
    """
    v = u.copy()
    for i in range(1, len(v)):
        if v[i] <= v[i-1]:
            v[i] = v[i-1] + 1e-16
    return v

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

def diff2_dinterp(A_, x):
    """ Perform row-wise differences in array A. Handle NaNs by extrapolating
    differences downward, performing a centred differences, and replacing the
    NaN values in the output.

    Compared to diff2 and diff2_inner, this is a generally much less accurate
    approximation (L-∞), however it has the virtue of allowing centred
    differences everywhere, avoiding non-physical jumps in the resulting field.
    """

    def erode_nans(u, du):
        """ Given u and a derivative du, extend u whereever a NaN borders a non-NaN """
        isnan = np.isnan
        for j in range(len(u)):

            if isnan (u[j]) and j != 0 and j != len(u)-1:
                if isnan(u[j+1]) and not isnan(u[j-1]):
                    d = 0.5 * (du[j] + du[j-1])
                    u[j] = u[j-1] + d
                elif not isnan(u[j+1]) and isnan(u[j-1]):
                    d = 0.5 * (du[j] + du[j+1])
                    u[j] = u[j+1] - d
                elif not isnan(u[j+1]) and not isnan(u[j-2]):
                    #u[j] = 0.5 * (u[j-1] + u[j+1])
                    u[j] = 0.5 * (u[j-1] + 0.5 * (du[j] + du[j-1]) +
                                  u[j+1] - 0.5 * (du[j] + du[j+1]))
                else:
                    pass

            elif j == 0 and isnan(u[j]) and not isnan(u[j+1]):
                u[j] = u[j+1] - 0.5 * (du[j] + du[j+1])

            elif j == len(u)-1 and isnan(u[j]) and not isnan(u[j-1]):
                u[j] = u[j-1] + 0.5 * (du[j] + du[j-1])

            else:
                pass

        return u

    A = A_.copy()
    D2 = np.empty_like(A_)
    for (i, row) in enumerate(A):

        if np.any(np.isnan(row)):

            if not np.all(np.isnan(row)) and i != 0:

                while np.any(np.isnan(row)):
                    row = erode_nans(row, D2[i-1,:])
                A[i,:] = row

            else:

                # Extend upward
                for j in range(len(row)):

                    if np.isnan(row[j]):
                        k = 0
                        while k != A.shape[0]-1:
                            if not np.isnan(A[k,j]):
                                A[:k,j] = A[k,j]
                                break
                            k += 1

                        if k == A.shape[0]-1:
                            raise ValueError("there's a whole column of NaNs")

        D2[i,:] = diff1(A[i,:], x)

        if i != 0:                      # Replace differences with the previous row
            nans = np.isnan(A_[i,:])    # to avoid propagating synthetic data
            nans = binary_dilation(nans)
            D2[i,nans] = D2[i-1,nans]

    D2[np.isnan(A_)] = np.nan
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
