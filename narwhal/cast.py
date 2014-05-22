"""
Cast and CastCollection classes for managing CTD observations
"""

import sys
import itertools
import collections
import json
import numbers
import numpy as np
import scipy.integrate as scint 
from karta import Point, LONLAT
from . import fileio
from . import gsw

class Cast(object):
    """ A Cast is a set of referenced measurements associated with a single
    coordinate.
    
    Water properties are provided as keyword arguments. There are several
    reserved keywords:

    *coords*        Tuple containing the geographic coordinates of the
                    observation

    *properties*    Dictionary of scalar metadata

    *primarykey*    Indicates the name of vertical measure. Usually pressure
                    ("pres"), but could be other things, e.g. depth ("z")
    """

    _type = "cast"

    def __init__(self, p, coords=None, properties=None, primarykey="pres",
                 **kwargs):

        if properties is None:
            self.properties = {}
        elif isinstance(properties, dict):
            self.properties = properties
        else:
            raise TypeError("properties must be a dictionary")

        self.primarykey = primarykey
        self.coords = coords
        self.data = collections.OrderedDict()
        self.data[primarykey] = np.asarray(p)

        def _fieldmaker(n, arg):
            if arg is not None:
                return np.asarray(arg)
            else:
                return np.nan * np.empty(n)

        # Python 3 workaround
        try:
            items = kwargs.iteritems()
        except AttributeError:
            items = kwargs.items()

        for (kw, val) in items:
            self.data[kw] = _fieldmaker(len(p), val)

        self._len = len(p)
        self._fields = tuple([primarykey] + [a for a in kwargs])
        return

    def __len__(self):
        return self._len

    def __str__(self):
        if self.coords is not None:
            coords = tuple(round(c, 3) for c in self.coords)
        else:
            coords = (None, None)
        s = "CTD cast (" + "".join([str(k)+", " for k in self._fields])
        # cut off the final comma
        s = s[:-2] + ") at {0}".format(coords)
        return s

    def __getitem__(self, key):
        if isinstance(key, int):
            if key < self._len:
                return tuple((a, self.data[a][key]) for a in self._fields
                             if hasattr(self.data[a], "__iter__"))
            else:
                raise IndexError("Index ({0}) is greater than cast length "
                                 "({1})".format(key, self._len))
        elif key in self.data:
            return self.data[key]
        else:
            raise KeyError("No item {0}".format(key))
        return

    def __setitem__(self, key, val):
        if isinstance(key, str):
            self.data[key] = val
        elif isinstance(key, int):
            raise KeyError("Cast object profiles are not mutable")
        else:
            raise KeyError("Cannot use {0} as a hash".format(key))
        return

    def __add__(self, other):
        if hasattr(other, "_type") and (other._type == "cast"):
            return CastCollection(self, other)
        elif hasattr(other, "_type") and (other._type == "ctd_collection"):
            return CastCollection(self, *[a for a in other])
        else:
            raise TypeError("No rule to add {0} to {1}".format(type(self), 
                                                               type(other)))

    def __eq__(self, other):
        if self._fields != other._fields or \
                self.properties != other.properties or \
                self.coords != other.coords or \
                False in (np.all(self.data[k] == other.data[k])
                                for k in self._fields):
            return False
        else:
            return True

    def nanmask(self):
        """ Return a mask for observations containing at least one NaN. """
        vectors = [a for a in self.data.values() if hasattr(a, "__iter__")]
        return np.isnan(np.vstack(vectors).sum(axis=0))

    def nvalid(self):
        """ Return the number of complete (non-NaN) observations. """
        vectors = [a for a in self.data.values() if hasattr(a, "__iter__")]
        nv = sum(reduce(lambda a,b: (~np.isnan(a))&(~np.isnan(b)), vectors))
        return nv

    def interpolate(self, y, x, v, force=False):
        """ Interpolate property y as a function of property x at values given by vector x=v.

        y::string       name of property to interpolate
        x::string       name of reference property
        v::iterable     vector of values for x

        force::bool     whether to coerce x to be monotonic (defualt False)

        Note: it's difficult to interpolate when x is not monotic, because this
        makes y not a true function. However, it's resonable to want to
        interpolate using rho or sigma as x. These should be essentially
        monotonic, but might not be due to measurement noise. The keyword
        argument `force` can be provided as True, which causes nonmonotonic x
        to be coerced into a monotonic form (see `force_monotonic`).
        """
        if y not in self.data:
            raise KeyError("Cast has no property '{0}'".format(y))
        elif x not in self.data:
            raise KeyError("Cast has no property '{0}'".format(x))
        if np.all(np.diff(self[x]) > 0.0):
            return np.interp(v, self[x], self[y])
        elif force:
            return np.interp(v, force_monotonic(self[x]), self[y])
        else:
            raise ValueError("x is not monotonic")

    def save(self, fnm):
        """ Save a JSON-formatted representation to a file.

        fnm::string     File name to save to
        """
        with open(fnm, "w") as f:
            fileio.writecast(f, self)
        return


class CTDCast(Cast):
    """ Specialization of Cast guaranteed to have salinity and temperature
    fields. """
    _type = "ctdcast"

    def __init__(self, p, sal=None, temp=None, coords=None, properties=None,
                 **kwargs):
        super(CTDCast, self).__init__(p, sal=sal, temp=temp, coords=coords,
                                      properties=properties, **kwargs)
        return

class LADCP(Cast):
    """ Specialization of Cast for LADCP data. Requires *u* and *v* fields. """
    _type = "ladcpcast"

    def __init__(self, z, u=None, v=None, err=None, coords=None, properties=None,
                 **kwargs):
        super(LADCP, self).__init__(z, u=u, v=v, err=err, coords=coords,
                                    properties=properties, **kwargs)
        return


class XBTCast(Cast):
    """ Specialization of Cast with temperature field. """
    _type = "xbtcast"

    def __init__(self, p, temp=None, coords=None, properties=None, **kwargs):
        super(XBTCast, self).__init__(p, temp=temp, coords=coords,
                                      properties=properties, **kwargs)
        return


class CastCollection(collections.Sequence):
    """ A CastCollection is an indexable collection of Cast instances """
    _type = "ctd_collection"

    def __init__(self, *args):
        if len(args) == 0:
            self.casts = []
        elif isinstance(args[0], Cast):
            self.casts = list(args)
        elif (len(args) == 1) and (False not in (isinstance(a, Cast) for a in args[0])):
            self.casts = args[0]
        else:
            raise TypeError("Arguments must be either Cast types or an "
                            "iterable collection of Cast types")
        return

    def __len__(self):
        return len(self.casts)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.casts.__getitem__(key)
        elif isinstance(key, slice):
            return type(self)(self.casts.__getitem__(key))
        elif False not in (key in cast.data for cast in self.casts):
            return np.vstack([a[key] for a in self.casts]).T
        else:
            raise KeyError("Key {0} not found in all casts".format(key))

    def __contains__(self, cast):
        return True if (cast in self.casts) else False

    def __iter__(self):
        return (a for a in self.casts)

    def __add__(self, other):
        if hasattr(other, "_type") and (other._type == "ctd_collection"):
            return CastCollection(list(a for a in itertools.chain(self.casts, other.casts)))
        elif hasattr(other, "_type") and (other._type == "cast"):
            return CastCollection(self.casts + [other])
        else:
            raise TypeError("Addition requires both arguments to fulfill the "
                            "ctd_collection interface")

    def add_bathymetry(self, bathymetry):
        """ Reference Bathymetry instance `bathymetry` to CastCollection.

        bathymetry::Bathymetry2d        bathymetry instance
        """
        for cast in self.casts:
            if hasattr(cast, "coords"):
                cast.properties["depth"] = bathymetry.atxy(*cast.coords)
            else:
                cast.properties["tdepth"] = np.nan
                sys.stderr.write("Warning: cast has no coordinates")
        return

    def mean(self):
        raise NotImplementedError()

    def asarray(self, key):
        """ Naively return values as an array, assuming that all casts are indexed
        with the same pressure levels.

        key::string         property to return
        """
        nrows = max(cast._len for cast in self.casts)
        arr = np.nan * np.empty((nrows, len(self.casts)), dtype=np.float64)
        for i, cast in enumerate(self.casts):
            arr[:cast._len, i] = cast[key]
        return arr

    def projdist(self):
        """ Return the cumulative distances from the cast to cast.
        """
        cumulative = [0]
        a = Point(self.casts[0].coords, crs=LONLAT)
        for cast in self.casts[1:]:
            b = Point(cast.coords, crs=LONLAT)
            cumulative.append(cumulative[-1] + a.distance(b))
            a = b
        return np.asarray(cumulative, dtype=np.float64)

    def thermal_wind(self, tempkey="temp", salkey="sal", rhokey=None,
                     dudzkey="dUdz", ukey="U", overwrite=False):
        """ Compute profile-orthagonal velocity shear using hydrostatic thermal
        wind. In-situ density is computed from temperature and salinity unless
        *rhokey* is provided.

        Add a U field and a ∂U/∂z field to each cast in the collection.
        
        Parameters
        ----------
        tempkey         key to use for temperature if *rhokey* is None
        salkey          key to use for salinity if *rhokey* is None
        rhokey          key to use for density
        dudzkey         key to use for ∂U/∂z, subject to *overwrite*
        ukey            key to use for U, subject to *overwrite*
        overwrite       whether to allow cast fields to be overwritten
                        if False, then *ukey* and *dudzkey* are incremented
                        until there is no clash
        """

        if rhokey is None:
            Temp = self.asarray(tempkey)
            Sal = self.asarray(salkey)
            Pres = self.asarray("pres")
            rho = np.empty_like(Pres)
            (m, n) = rho.shape
            for i in range(m):
                for j in range(n):
                    ct = gsw.ct_from_t(Sal[i,j], Temp[i,j], Pres[i,j])
                    rho[i,j] = gsw.rho(Sal[i,j], ct, Pres[i,j])
            del Temp
            del Sal
            del Pres
        else:
            rho = self.asarray(rhokey)
            (m, n) = rho.shape

        g = 9.8
        omega = 2*np.pi / 86400.0
        drho = diff2(rho, self.projdist())
        dUdz = -(g / rho * drho) / (2*omega)
        U = uintegrate(dUdz, self.asarray("pres"))

        dudzkey_ = dudzkey
        ukey_ = ukey
        for (ic,cast) in enumerate(self.casts):
            if not overwrite:
                i = 2
                while dudzkey_ in cast.data:
                    dudzkey_ = dudzkey + "_" + str(i)
                    i += 1
                i = 2
                while ukey_ in cast.data:
                    ukey_ = ukey + "_" + str(i)
                    i += 1
            cast.data[dudzkey_] = dUdz[:,ic]
            cast.data[ukey_] = U[:,ic]
        return

    def save(self, fnm):
        """ Save a JSON-formatted representation to a file.

        fnm::string     File name to save to
        """
        with open(fnm, "w") as f:
            fileio.writecastcollection(f, self)



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

def read(fnm):
    """ Convenience function for reading JSON-formatted measurement data. """
    with open(fnm, "r") as f:
        d = json.load(f)
    typ = d.get("type", None)
    if typ == "cast":
        return fileio.dictascast(d, Cast)
    elif typ == "ctdcast":
        return fileio.dictascast(d, CTDCast)
    elif typ == "xbtcast":
        return fileio.dictascast(d, XBTCast)
    elif typ == "ladcpcast":
        return fileio.dictascast(d, LADCP)
    elif typ == "ctd_collection":
        return CastCollection(fileio.dictascastcollection(d, Cast))
    elif typ is None:
        raise IOError("couldn't read data type - file may be corrupt")
    else:
        raise IOError("Invalid type: {0}".format(typ))

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
                else:
                    assert j-start == 1    # if this isn't true, I screwed up somewhere
                start = -1
            elif start != -1 and j == len(arow) - 1:
                D2[i,start:] = diff1(arow[start:], x[start:])
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
