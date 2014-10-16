""" Built-in interpolation functions for plotting fields on observations. """

import numpy as np
from scipy.interpolate import griddata, CloughTocher2DInterpolator

def cubic(X, Y, Z, Xi, Yi):
    """ Wraps scipy's cubic spline function. """
    X = np.array(X)
    Y = np.array(Y)
    Z = np.array(Z)
    msk = ~np.isnan(X+Y+Z)
    Zi = griddata(np.c_[X[msk], Y[msk]], Z[msk], np.c_[Xi.ravel(), Yi.ravel()],
                  method="cubic")
    Zi = Zi.reshape(Xi.shape)
    return Xi, Yi, Zi

def horizontal_corr(X, Y, Z, Xi, Yi):
    # this version works with irregular gridding

    N = sum(np.sum(~np.isnan(z)) for x in Z)
    X_, Y_, Z_ = np.empty(N), np.empty(N), np.empty(N)
    d = X[0]

    for i,x in enumerate(d):
        # add the measured values
        y = np.array(Y[i])
        z = np.array(Z[i])
        valid = ~np.isnan(z)
        n = np.sum(valid)

        X_.extend(x * np.ones(n))
        Y_.extend(y[valid])
        Z_.extend(z[valid])

        # for each non-NaN obs in Z[i], check Z[i-1] and Z[i+1] to see if
        # they're NaN, and if so add a dummy value half way between
        for lvl, v in zip(y[valid], z[valid]):

            zname = cast.zname
            if i != 0 and np.isnan(np.interp(Y[i-1], Z[i-1], lvl)):
                X_.append(0.5 * (d[i] + d[i-1]))
                Y_.append(lvl)
                Z_.append(v)
        
            if i != len(d) and np.isnan(np.interp(Y[i+1], Z[i+1], lvl)):
                X_.append(0.5 * (d[i] + d[i+1]))
                Y_.append(lvl)
                Z_.append(v)

    X_ = np.array(X_)
    Y_ = np.array(Y_)
    ct = CloughTocher2DInterpolator(np.c_[0.0001*X_, Y_], Z_)
    Zi = ct(0.0001*Xi, Yi)
    return Zi

def zero_base(X, Y, Z, Zi, Yi)

    # Add zero boundary condition
    def _find_idepth(arr):
        idepth = []
        for col in arr.T:
            _inonnan = np.arange(len(col))[~np.isnan(col)]
            idepth.append(_inonnan[-1])
        return idepth

    idxdepth = _find_idepth(Z)
    idepth = np.round(np.interp(Xi[0], X[0], idxdepth)).astype(int)

    xbc, ybc, zbc = [], [], []
    for j, i in enumerate(idepth):
        xbc.append(xi[j])
        ybc.append(yi[i]+2)
        zbc.append(0.0)

    X = np.array(X)
    Y = np.array(Y)
    Z = np.array(Z)
    msk = ~np.isnan(X + Y + Z)

    X_bc = np.r_[X[msk], xbc]
    Y_bc = np.r_[Y[msk], ybc]
    Z_bc = np.r_[Z[msk], zbc]

    alpha = 1e4
    Zi = griddata(np.c_[X_bc, alpha*Y_bc], Z_bc,
                  np.c_[Xi.ravel(), alpha*Yi.ravel()], method="cubic")

    Zi = Zi.reshape(Xi.shape)

