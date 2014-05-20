"""
Cast and CastCollection classes for managing CTD observations
"""

import sys
import itertools
import collections
import json
import numpy as np
from karta import Point, LONLAT
from . import fileio
from . import gsw

class Cast(object):
    """ A Cast is a set of pressure-referenced measurements associated with a
    single coordinate. """

    _type = "cast"

    def __init__(self, p, coords=None, bathymetry=None, **kwargs):

        self.coords = coords
        self.bath = bathymetry

        self.data = collections.OrderedDict()
        self.data["pres"] = p

        def _fieldmaker(n, arg):
            if arg is not None:
                return arg
            else:
                return [None for _ in xrange(n)]

        # Python 3 workaround
        try:
            items = kwargs.iteritems()
        except AttributeError:
            items = kwargs.items()

        for kw, val in items:
            self.data[kw] = _fieldmaker(len(p), val)

        self._len = len(p)
        self._fields = tuple(["pres"] + [a for a in kwargs])
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
            raise TypeError("No rule to add {0} to {1}".format(type(self), type(other)))

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
    """ Specialization of Cast guaranteed to have salinity and temperature fields. """

    def __init__(self, p, sal=None, temp=None, coords=None, bathymetry=None,
                 **kwargs):
        super(CTDCast, self).__init__(p, sal=sal, temp=temp, coords=coords,
                                      bathymetry=bathymetry, **kwargs)
        return


class XBTCast(Cast):
    """ Specialization of Cast with temperature field. """

    def __init__(self, p, temp=None, coords=None, bathymetry=None, **kwargs):
        super(XBTCast, self).__init__(p, temp=temp, coords=coords,
                                      bathymetry=bathymetry, **kwargs)
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
                cast["botdepth"] = bathymetry.atxy(*cast.coords)
            else:
                cast["botdepth"] = np.nan
                sys.stderr.write("Warning: cast has no coordinates")
        self.bath = bathymetry
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
        return cumulative

    def thermal_wind(self, tempkey="temp", salkey="sal", rhokey=None):
        """ Compute profile-orthagonal relative velocity shear using hydrostatic
        thermal wind. Output is referenced to a bottom velocity of zero.
        """
        if rhokey is None:
            Temp = self.asarray(tempkey)
            Sal = self.asarray(salkey)
            Pres = self.asarray("pres")
            Rho = np.empty_like(Pres)
            (m, n) = Rho.shape
            for i in range(m):
                for j in range(n):
                    ct = gsw.ct_from_t(Sal[i,j], Temp[i,j], Pres[i,j])
                    Rho[i,j] = gsw.rho(Sal[i,j], ct, Pres[i,j])
            del Temp
            del Sal
            del Pres
        else:
            Rho = self.asarray(rhokey)
            (m, n) = Rho.shape

        g = 9.8
        omega = 2*np.pi / 86400.0
        d = np.diff(self.projdist())
        dRho = np.empty_like(Rho)
        dRho = np.hstack([np.atleast_2d(dRho[:,1] - dRho[:,0]).T / d[0],
                          (dRho[:,2:] - dRho[:,:-2]) / np.diff(d),
                          np.atleast_2d(dRho[:,-1] - dRho[:,-2]).T / d[-1]])
        dUdz = -0.5 * (g / Rho * dRho) / omega
        return dUdz

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
    if d.get("type", None) == "cast":
        return fileio.dictascast(d, Cast)
    elif d.get("type", None) == "ctd_collection":
        return CastCollection(fileio.dictascastcollection(d, Cast))
    else:
        raise IOError("Invalid input file")

