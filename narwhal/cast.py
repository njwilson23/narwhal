import sys
import itertools
import collections
import numpy as np

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
                return tuple(self.data[a][key] for a in self._fields)
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
            raise KeyError("Cast object profiles are not intended to be "
                           "mutable")
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

    def interpolate(self, y, x, v):
        """ Interpolate y as a function of x at x=v.. """
        if y not in self.data:
            raise KeyError("Cast has no property '{0}'".format(y))
        elif x not in self.data:
            raise KeyError("Cast has no property '{0}'".format(x))
        if not np.all(np.diff(self[x]) > 0.0):
            raise NotImplementedError("does not yet handle interpolation of "
                                      "f(x) where x is non-monotonic")
        return np.interp(v, self[x], self[y])


class CastCollection(collections.Sequence):
    """ A CastCollection is an indexable collection of Cast instances """
    _type = "ctd_collection"

    def __init__(self, *args):
        if isinstance(args[0], Cast):
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
            return np.hstack([a[key] for a in self.casts])
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
        """
        for cast in self.casts:
            if hasattr(cast, "coords"):
                cast["botdepth"] = bathymetry.at(self.coords)
            else:
                cast["botdepth"] = np.nan
                sys.stderr.write("Warning: cast has no coordinates")
        self.bath = bathymetry
        return

    def mean(self):
        raise NotImplementedError()

    def asarray(self, key):
        """ Naively return values as an array, assuming that all casts are indexed
        with the same pressure levels """
        nrows = max(cast._len for cast in self.casts)
        arr = np.nan * np.empty((nrows, len(self.casts)), dtype=np.float64)
        for i, cast in enumerate(self.casts):
            arr[:cast._len, i] = cast[key]
        return arr

    def versus(self, key1, key2):
        """ Return arrays containing bulk values from all casts. The values are
        determined by `key1` and `key2`. """
        v1 = self[key1]
        v2 = self[key2]
        return v1, v2

