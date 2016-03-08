# -*- coding: utf-8 -*-
""" Analysis functions for casts and collections of casts """

from functools import reduce

import numpy as np

from .cast import Cast, CastCollection
from .cast import NarwhalError, FieldError

from . import util
import scipy.sparse
import scipy.sparse.linalg

# Global physical constants
G = 9.81
OMEGA = 2*np.pi / 86400.0

def water_fractions(cast, sources, tracers=("salinity", "temperature")):
    """ Compute water mass fractions based on *n* (>= 2) conservative
    tracers.

    Arguments
    ---------

    sources (list of tuples)

        List of *n+1* tuples specifying prototype water masses in terms of
        `tracers` (see below). Each tuple must have length *n*.

    tracers (list of strings)

        *n* strings serving as Cast fields to use as tracers [default:
        ("salinity", "temperature")].
    """
    n = len(tracers)
    if n < 2:
        raise NarwhalError("Two or more prototype waters required")

    m = cast.nvalid(tracers)
    I = scipy.sparse.eye(m)
    A_ = np.array([[src[i] for src in sources] for i in range(n)])
    A = np.vstack([A_, np.ones(n+1, dtype=np.float64)])
    As = scipy.sparse.kron(I, A, "csr")
    b = np.zeros((n+1)*m)
    msk = cast.nanmask(tracers)
    for i in range(n):
        b[i::n+1] = cast[tracers[i]][~msk]
    b[n::n+1] = 1.0             # lagrange multiplier

    frac = scipy.sparse.linalg.spsolve(As, b)
    chis = [np.empty(len(cast)) * np.nan for i in range(n+1)]
    for i in range(n+1):
        chis[i][~msk] = frac[i::n+1]
    return chis


def baroclinic_modes(cast, nmodes, ztop=10, N2key="N2", depthkey="depth"):
    """ Calculate the baroclinic normal modes based on linear
    quasigeostrophy and the vertical stratification. Return the first
    *nmodes* deformation radii and their associated eigenfunctions.

    Currently requires regular vertical gridding in depth.

    Args
    ----
    cast (Cast): profile to base normal modes on
    nmodes (int): number of modes to compute
    ztop (float): depth to cut off the profile, to avoid surface effects
    N2key (str): data key to use for buoyancy frequency N^2
    depthkey (str): data key to use for depth
    """
    if (N2key not in cast.fields) or (depthkey not in cast.fields):
        raise FieldError("buoyancy frequency and depth required")

    igood = ~cast.nanmask((N2key, depthkey))
    N2 = cast[N2key][igood]
    dep = cast[depthkey][igood]

    itop = np.argwhere(dep > ztop)[0]
    N2 = N2[itop:].values
    dep = dep[itop:].values

    h = np.diff(dep)
    assert all(h[0] == h_ for h_ in h[1:])      # requires uniform gridding

    f = 2*OMEGA * np.sin(cast.coordinates.y)
    F = f**2/N2
    F[0] = 0.0
    F[-1] = 0.0
    F = scipy.sparse.diags(F, 0)

    D1 = util.sparse_diffmat(len(N2), 1, h[0])
    D2 = util.sparse_diffmat(len(N2), 2, h[0])

    T = scipy.sparse.diags(D1 * F.diagonal(), 0)
    M = T*D1 + F*D2
    lamda, V = scipy.sparse.linalg.eigs(M.tocsc(), k=nmodes+1, sigma=1e-8)
    Ld = 1.0 / np.sqrt(np.abs(np.real(lamda[1:])))
    return Ld, V[:,1:]


def thermal_wind(castcoll, rhokey="density", depthkey="depth"):
    """ Compute profile-orthagonal velocity shear using hydrostatic thermal
    wind.

    Args
    ----
    rhokey (str): key to use for density

    depthkey (str): key to use for depth from the surface

    Returns
    -------
    CastCollection containing casts with shear and velocity fields
    """
    if not all(rhokey in c.fields for c in castcoll):
        raise FieldError("Not all casts have a `{0}` field".format(rhokey))

    if not all(depthkey in c.fields for c in castcoll):
        raise FieldError("Not all casts have a `{0}` field".format(depthkey))

    rho = castcoll.asarray(rhokey)
    (m, n) = rho.shape

    drho = util.diff2_dinterp(rho, castcoll.projdist())
    sinphi = np.sin([c.coordinates.y*np.pi/180.0 for c in castcoll.casts])
    dudz = (G / rho * drho) / (2*OMEGA*sinphi)
    u = util.uintegrate(dudz, castcoll.asarray(depthkey))

    outcasts = []
    for i in range(len(castcoll)):
        outcasts.append(Cast(dudz=dudz[:,i],
                             uvel=u[:,i],
                             coordinates=castcoll[i].coordinates.vertex))
    return CastCollection(outcasts)

def thermal_wind_inner(castcoll, tempkey="temperature", salkey="salinity",
                       rhokey=None, depthkey="depth", dudzkey="dudz", ukey="u",
                       bottomkey="bottom", overwrite=False):
    """ Alternative implementation that creates a new cast collection
    consistng of points between the observation casts.

    Compute profile-orthagonal velocity shear using hydrostatic thermal wind.

    Args
    ----
    rhokey (str): key to use for density

    depthkey (str): key to use for depth from the surface

    Returns
    -------
    CastCollection containing casts with shear and velocity fields
    """
    if not all(rhokey in c.fields for c in castcoll):
        raise FieldError("Not all casts have a `{0}` field".format(rhokey))

    if not all(depthkey in c.fields for c in castcoll):
        raise FieldError("Not all casts have a `{0}` field".format(depthkey))

    rho = castcoll.asarray(rhokey)
    (m, n) = rho.shape

    def avgcolumns(a, b):
        avg = a if len(a[~np.isnan(a)]) > len(b[~np.isnan(b)]) else b
        return avg

    # compute mid locations
    midcoords = []
    for i in range(len(castcoll)-1):
        c1 = castcoll[i].coordinates
        c2 = castcoll[i+1].coordinates
        az, _, d = c1.crs.inverse(c1.x, c1.y, c2.x, c2.y)
        x, y, _ = c1.crs.forward(c1.x, c1.y, az, 0.5*d)
        midcoords.append((x, y))

    drho = util.diff2_inner(rho, castcoll.projdist())
    sinphi = np.sin([c[1]*np.pi/180.0 for c in midcoords])
    rhoavg = 0.5 * (rho[:,:-1] + rho[:,1:])
    dudz = (G / rhoavg * drho) / (2*OMEGA*sinphi)
    u = util.uintegrate(dudz, castcoll.asarray(depthkey))

    outcasts = []
    for i, (x, y) in enumerate(midcoords):
        outcasts.append(Cast(dudz=dudz[:,i], uvel=u[:,i],
                             coordinates=(x, y)))
    return CastCollection(outcasts)

def eofs(castcoll, key="temperature", zkey="depth", n_eofs=None):
    """ Compute the EOFs and EOF structures for *key*.

    Requires all casts to have the same depth-gridding.

    Args
    ----
    key (str): key to use for computing EOFs
    n_eofs (int): number of EOFs to return

    Returns
    -------
    Cast containing the structure functions as fields
    ndarray containing eigenvectors
    ndarray containing eigenvalues
    """
    assert all(zkey in c.fields for c in castcoll)
    assert all(all(castcoll[0].data.index == c.data.index) for c in castcoll[1:])

    if n_eofs is None:
        n_eofs = len(castcoll)

    arr = castcoll.asarray(key)
    msk = reduce(lambda a,b:a|b, (c.nanmask(key) for c in castcoll))
    arr = arr[~msk,:]
    arr -= arr.mean()

    _, sigma, V = np.linalg.svd(arr)
    lamb = sigma**2/(len(castcoll)-1)
    eofts = util.eof_timeseries(arr, V)

    c0 = castcoll[0]
    cast = Cast(**{zkey: c0[zkey][~msk]})
    for i in range(n_eofs):
        cast._addkeydata("_eof".join([key, str(i+1)]), eofts[:,i])
    return cast, lamb[:n_eofs], V[:,:n_eofs]
