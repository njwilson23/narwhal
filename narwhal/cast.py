"""
Cast and CastCollection classes for managing CTD observations
"""

import sys
import itertools
import collections
import numpy as np
from karta import Point, LONLAT

class Cast(object):
    """ A Cast is a set of pressure, salinity, temperature (et al)
    measurements associated with a single coordinate. """

    _type = "ctd_cast"

    def __init__(self, p, S=None, T=None, coords=None, bathymetry=None,
        **kwargs):

        self.coords = coords
        self.bath = bathymetry

        def _fieldmaker(n, arg):
            if arg is not None:
                return arg
            else:
                return [None for _ in xrange(n)]

        self.data = {}
        self.data["pres"] = p
        self.data["sal"] = _fieldmaker(len(p), S)
        self.data["temp"] = _fieldmaker(len(p), T)

        for kw,val in kwargs.iteritems():
            self.data[kw] = _fieldmaker(len(p), val)

        self._len = len(p)
        self._fields = tuple(["pres", "sal", "temp"] + [a for a in kwargs])
        return

    def __repr__(self):
        return "CTD cast <{0}> at {1}".format(self._fields, self.coords)

    def __getitem__(self, key):
        if isinstance(key, int):
            if key < self._len:
                return tuple(self.data[a][key] for a in self._fields
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
        if hasattr(other, "_type") and (other._type == "ctd_cast"):
            return CastCollection(self, other)
        elif hasattr(other, "_type") and (other._type == "ctd_collection"):
            return CastCollection(self, *[a for a in other])
        else:
            raise TypeError("No rule to add {0} to {1}".format(type(self), type(other)))

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
        elif hasattr(other, "_type") and (other._type == "ctd_cast"):
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

