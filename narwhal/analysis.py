""" Analysis functions for casts and collections of casts """

from functools import reduce

import numpy as np

from .cast import Cast, CastCollection
from .cast import CTDCast
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
    `nmodes::int` deformation radii and their associated eigenfunctions.

    Arguments
    ---------

    ztop (float)

        the depth at which to cut off the profile, to avoid surface effects

    N2key (string)

        data key to use for N^2

    depthkey (string)

        data key to use for depth
    """
    if N2key not in cast.fields or depthkey not in cast.fields:
        raise FieldError("buoyancy frequency and depth required")

    igood = ~cast.nanmask((N2key, depthkey))
    N2 = cast[N2key][igood]
    dep = cast[depthkey][igood]

    itop = np.argwhere(dep > ztop)[0]
    N2 = N2[itop:].values
    dep = dep[itop:].values

    h = np.diff(dep)
    assert all(h[0] == h_ for h_ in h[1:])     # requires uniform gridding for now

    f = 2*OMEGA * np.sin(cast.coords[1])
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


def thermal_wind(castcoll, tempkey="temperature", salkey="salinity",
                 rhokey=None, depthkey="depth", dudzkey="dudz", ukey="u", 
                 overwrite=False):
    """ Compute profile-orthagonal velocity shear using hydrostatic thermal
    wind. In-situ density is computed from temperature and salinity unless
    *rhokey* is provided.

    Adds a U field and a ∂U/∂z field to each cast in the collection. As a
    side-effect, if casts have no "depth" field, one is added and populated
    from temperature and salinity fields.

    Arguments
    ---------

    tempkey (string)

        key to use for temperature if *rhokey* is None

    salkey (string)

        key to use for salinity if *rhokey* is None

    rhokey (string)

        key to use for density, or None [default: None]

    dudzkey (string)

        key to use for ∂U/∂z, subject to *overwrite*

    ukey (string)

        key to use for U, subject to *overwrite*

    overwrite (bool)

        whether to allow cast fields to be overwritten if False, then
        *ukey* and *dudzkey* are incremented until there is no clash
    """
    if rhokey is None:
        rhokeys = []
        for cast in castcoll.casts:
            rhokeys.append(cast.add_density())
        if any(r != rhokeys[0] for r in rhokeys[1:]):
            raise NarwhalError("Tried to add density field, but got inconsistent keys")
        else:
            rhokey = rhokeys[0]

    rho = castcoll.asarray(rhokey)
    (m, n) = rho.shape

    for cast in castcoll:
        if depthkey not in cast.data.keys():
            cast.add_depth()

    drho = util.diff2_dinterp(rho, castcoll.projdist())
    sinphi = np.sin([c.coords[1]*np.pi/180.0 for c in castcoll.casts])
    dudz = (G / rho * drho) / (2*OMEGA*sinphi)
    u = util.uintegrate(dudz, castcoll.asarray(depthkey))

    for (ic,cast) in enumerate(castcoll.casts):
        cast._addkeydata(dudzkey, dudz[:,ic], overwrite=overwrite)
        cast._addkeydata(ukey, u[:,ic], overwrite=overwrite)
    return

def thermal_wind_inner(castcoll, tempkey="temperature", salkey="salinity",
                       rhokey=None, depthkey="depth", dudzkey="dudz", ukey="u", 
                       bottomkey="bottom", overwrite=False):
    """ Alternative implementation that creates a new cast collection
    consistng of points between the observation casts.

    Compute profile-orthagonal velocity shear using hydrostatic thermal
    wind. In-situ density is computed from temperature and salinity unless
    *rhokey* is provided.

    Adds a U field and a ∂U/∂z field to each cast in the collection. As a
    side-effect, if casts have no "depth" field, one is added and populated
    from temperature and salinity fields.

    Arguments
    ---------

    tempkey (string)

        key to use for temperature if *rhokey* is None

    salkey (string)

        key to use for salinity if *rhokey* is None

    rhokey (string)

        key to use for density, or None [default: None]

    dudzkey (string)

        key to use for ∂U/∂z, subject to *overwrite*

    ukey (string)

        key to use for U, subject to *overwrite*

    overwrite (bool)

        whether to allow cast fields to be overwritten if False, then
        *ukey* and *dudzkey* are incremented until there is no clash
    """
    if rhokey is None:
        rhokeys = []
        for cast in castcoll.casts:
            rhokeys.append(cast.add_density())
        if any(r != rhokeys[0] for r in rhokeys[1:]):
            raise NarwhalError("Tried to add density field, but found inconsistent keys")
        else:
            rhokey = rhokeys[0]

    rho = castcoll.asarray(rhokey)
    (m, n) = rho.shape

    def avgcolumns(a, b):
        avg = a if len(a[~np.isnan(a)]) > len(b[~np.isnan(b)]) else b
        return avg

    # Add casts in intermediate positions
    midcasts = []
    for i in range(len(castcoll)-1):
        c1 = castcoll[i].coords
        c2 = castcoll[i+1].coords
        cmid = (0.5*(c1[0]+c2[0]), 0.5*(c1[1]+c2[1]))
        p = avgcolumns(castcoll[i]["pressure"], castcoll[i+1]["pressure"])
        t = avgcolumns(castcoll[i]["temperature"], castcoll[i+1]["temperature"])
        s = avgcolumns(castcoll[i]["salinity"], castcoll[i+1]["salinity"])
        cast = CTDCast(p, s, t, coords=cmid)
        if "depth" not in cast.fields:
            cast.add_density()
        cast.add_depth()
        cast.properties[bottomkey] = 0.5 * (castcoll[i].properties[bottomkey] +
                                            castcoll[i+1].properties[bottomkey])
        midcasts.append(cast)

    coll = CastCollection(midcasts)
    drho = util.diff2_inner(rho, castcoll.projdist())
    sinphi = np.sin([c.coords[1]*np.pi/180.0 for c in midcasts])
    rhoavg = 0.5 * (rho[:,:-1] + rho[:,1:])
    dudz = (G / rhoavg * drho) / (2*OMEGA*sinphi)
    u = util.uintegrate(dudz, coll.asarray(depthkey))

    for (ic,cast) in enumerate(coll):
        cast._addkeydata(dudzkey, dudz[:,ic], overwrite=overwrite)
        cast._addkeydata(ukey, u[:,ic], overwrite=overwrite)
    return coll

def eofs(castcoll, key="temperature", zkey="depth", n_eofs=None):
    """ Compute the EOFs and EOF structures for *key*. Returns a cast with
    the structure functions, an array of eigenvectors (EOFs), and an array
    of eigenvalues.

    Requires all casts to have the same depth-gridding.
    
    Arguments
    ---------

    key (string)

        key to use for computing EOFs

    n_eofs (int)

        number of EOFs to return
    """
    assert all(zkey in c.fields for c in castcoll)
    assert all(all(castcoll[0].data.index == c.data.index) for c in castcoll[1:])

    if n_eofs is None:
        n_eofs = len(castcoll)

    arr = castcoll.asarray(key)
    msk = reduce(lambda a,b:a|b, (c.nanmask(key) for c in castcoll))
    arr = arr[~msk,:]
    arr -= arr.mean()

    _,sigma,V = np.linalg.svd(arr)
    lamb = sigma**2/(len(castcoll)-1)
    eofts = util.eof_timeseries(arr, V)

    c0 = castcoll[0]
    cast = Cast(**{zkey: c0[zkey][~msk]})
    for i in range(n_eofs):
        cast._addkeydata("_eof".join([key, str(i+1)]), eofts[:,i])
    return cast, lamb[:n_eofs], V[:,:n_eofs]

