# -*- coding: utf-8 -*-
"""
Cast and CastCollection classes for managing CTD observations

Casts are a wrapper around a
pandas.Dataframe, with methods that are
useful to oceanographers.

Narwhal objects serialize to Python
dictionaries, with the following schema:

Dictionary schema:

{ __schemaversion__ ->  float: *version, currently 2.0*,
  type              ->  str: *type*,
  data              ->  { [key]?        ->  *value*,
                          [key]?        ->  *value* }
  properties        ->  { coordinates   ->  (float: *lon*, float: *lat*),
                          date|time?    ->  str: datetime (ISO formatted),
                          [key]?        ->  *value*,
                          [key]?        ->  *value* }
"""

import os
import sys
import abc
import collections
import itertools
import copy
import datetime, dateutil
from functools import reduce

import six
import numpy as np
import pandas

import scipy.ndimage
import scipy.io
import scipy.interpolate

try:
    from karta import Point, Multipoint
    from karta.crs import LonLatWGS84
except ImportError:
    from .geo import Point, Multipoint, LonLatWGS84

from . import gsw
from . import util
from . import iojson
from . import iohdf

class NarwhalBase(object):
    """ Base class for Narwhal objects implementing data export methods.

    Derived subclasses must implement `asdict`
    """

    def save_json(self, fnm, binary=True):
        """ Save a JSON-formatted representation to a file at `fnm::string`.
        """
        if binary:
            iojson.write_binary(fnm, self.asdict())
        else:
            iojson.write_text(fnm, self.asdict())
        return

    def save_hdf(self, fnm):
        return iohdf.write(fnm, self.asdict())

class Cast(NarwhalBase):
    """ A Cast is a set of referenced measurements associated with a single
    location.

    Args:

        Vector water properties and scalar metadata are provided as keyword
        arguments.

        There is one are several reserved keywords:

        coordinates (Optional[tuple]): Length 2 tuple providing the
            geographical location of a cast. If not provided, defaults to NaN.

        length (Optional[int]): Specifies the length of vector fields, which is
            used as a runtime check. If not provided, length is inferred from
            the other arguments.
    """

    _type = "cast"

    def __init__(self, length=None, **kwargs):

        self.properties = {}
        data = {}

        def isvec(a):
            return isinstance(a, collections.Container) and \
                    not isinstance(a, str)

        # Identify the profile length
        if length is None:
            length = max(len(v) for v in kwargs.values() if isvec(v))

        # Populate data and properties
        for (k, v) in kwargs.items():
            if isvec(v) and (len(v) == length):
                data[k] = pandas.Series(data=v, name=k)
            else:
                self.properties[k] = v

        self.data = pandas.DataFrame(data)
        self.properties.setdefault("coordinates", (np.nan, np.nan))
        return

    def __len__(self):
        return len(self.data.index)

    def __str__(self):
        if self.coordinates is not None:
            coords = tuple(round(c, 3) for c in self.coordinates)
        else:
            coords = (None, None)
        s = "Cast (" + "".join([str(k)+", " for k in self.fields])
        # cut off the final comma
        s = s[:-2] + ") at {0}".format(coords)
        return s

    def __getitem__(self, key):
        if isinstance(key, (int, np.int32, np.int64)):
            if 0 <= key < len(self):
                return self.data.irow(key)
            else:
                raise IndexError("{0} not within cast length ({1})".format(key, len(self)))
        elif key in self.data:
            return self.data[key]
        else:
            raise KeyError("No field {0}".format(key))
        return

    def __setitem__(self, key, val):
        if isinstance(key, str):
            if isinstance(val, collections.Container) and \
                    not isinstance(val, str) and len(val) == len(self):
                self.data[key] = val
                if key not in self.fields:
                    self.fields.append(key)
            else:
                raise NarwhalError("Fields must be set from iterables with length equal to the cast")

        elif isinstance(key, (int, np.int32, np.int64)):
            raise NarwhalError("Profiles are immutable")
        else:
            raise KeyError("{0} is an invalid key".format(key))
        return

    def __add__(self, other):
        if isinstance(other, AbstractCast):
            return CastCollection(self, other)
        elif isinstance(other, AbstractCastCollection):
            return CastCollection(self, *[a for a in other])
        else:
            raise TypeError("No rule to add {0} to {1}".format(type(self),
                                                               type(other)))

    def __eq__(self, other):
        if set(self.fields) != set(other.fields) or \
                self.properties != other.properties or \
                any(np.any(self.data[k] != other.data[k]) for k in self.fields):
            return False
        else:
            return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def _addkeydata(self, key, data, overwrite=False):
        """ Add `data::array` under `key::string`. If `key` already exists,
        iterates over [key]_2, [key]_3... until an unused identifier is found.
        Returns the key finally used.

        Use case: for automatic addition of fields.
        """
        key_ = key
        if not overwrite:
            i = 2
            while key_ in self.fields:
                key_ = key + "_" + str(i)
                i += 1
        ser = pandas.Series(data=data, index=self.data.index, name=key_)
        self.data = self.data.join(ser)
        return key_

    @property
    def p(self):
        return self.properties

    @property
    def fields(self):
        return list(self.data.columns)

    @property
    def coordinates(self):
        return Point(self.properties["coordinates"], crs=LonLatWGS84)

    def nanmask(self, fields=None):
        """ Return a mask for observations containing at least one NaN. """
        if fields is None:
            return self.data.isnull().apply(lambda r: any(r), axis=1).values
        else:
            return self.data.isnull().select(lambda n: n in fields, axis=1).apply(lambda r: any(r), axis=1).values

    def nvalid(self, fields=None):
        """ Return the number of complete (non-NaN) observations. """
        if fields is None:
            fields = self.fields
        elif isinstance(fields, str):
            fields = (fields,)
        vectors = [self.data[k] for k in fields]
        if len(vectors) == 1:
            nv = sum(~np.isnan(vectors[0]))
        else:
            nv = sum(reduce(lambda a,b: (~np.isnan(a))&(~np.isnan(b)), vectors))
        return nv

    def extend(self, n):
        """ Add *n* (int) NaN depth levels to cast. """
        if n > 0:
            d = dict((k, np.nan*np.empty(n)) for k in self.fields)
            self.data = pandas.concat([self.data, pandas.DataFrame(d)],
                                      ignore_index=True)
        else:
            raise NarwhalError("Cast must be extended with 1 or more rows")
        return

    def interpolate(self, y, x, v, force=False):
        """ Interpolate property y as a function of property x at values given
        by vector x=v.

        Args:

            y (str): name of property to interpolate
            x (str): name of reference property
            v (iterable): vector of values for x
            force (bool): whether to coerce x to be monotonic (default False)

        Note: it's difficult to interpolate when x is not monotonic, because
        this makes y not a true function. However, it's resonable to want to
        interpolate using rho or sigma as x. These should be essentially
        monotonic, but might not be due to measurement noise. The keyword
        argument `force` can be provided as True, which causes nonmonotonic x
        to be coerced into a monotonic form (see `force_monotonic`).
        """
        if y not in self.data:
            raise KeyError("Cast has no property '{0}'".format(y))
        elif x not in self.data:
            raise KeyError("Cast has no property '{0}'".format(x))
        dx = np.diff(self[x])
        if np.all(dx[~np.isnan(dx)] >= 0.0):
            return np.interp(v, self[x], self[y])
        elif force:
            return np.interp(v, util.force_monotonic(self[x]), self[y])
        else:
            raise NarwhalError("{x} is not monotonic; pass force=True to override".format(x=x))

    def regrid(self, levels, refkey):
        """ Re-interpolate Cast at specified grid levels. Returns a new Cast. """
        # some low level voodoo
        ret = copy.deepcopy(self)
        newdata = pandas.DataFrame(index=levels)
        for key in self.fields:
            newdata[key] = np.interp(levels, self.data[refkey], self[key],
                                     left=np.nan, right=np.nan)
        ret.data = newdata
        return ret

    def asdict(self):
        """ Return a representation of the Cast as a Python dictionary.
        """
        d = dict(__schemaversion__=2.0,
                 data=dict(), properties=dict(), type="cast")

        for col in self.data.columns:
            d["data"][col] = list(self.data[col].values)

        for k, v in self.properties.items():
            if isinstance(v, datetime.datetime):
                d["properties"][k] = v.isoformat(sep=" ")
            elif isinstance(v, (datetime.time, datetime.date)):
                d["properties"][k] = v.isoformat()
            else:
                try:
                    d["properties"][k] = v
                except TypeError:
                    print("Unable to serialize property {0} = {1}".format(k, v))
        return d

    def add_density(self, salkey="salinity", tempkey="temperature",
                    preskey="pressure", rhokey="density"):
        """ Add in-situ density computed from salinity, temperature, and
        pressure to fields. Return the field name.

        Args:

            salkey (str): data key to use for salinity
            tempkey (str): data key to use for in-situ temperature
            preskey (str): data key to use for pressure
            rhokey (str): data key to use for in-situ density
        """
        if salkey in self.fields and tempkey in self.fields and preskey in self.fields:
            SA = gsw.sa_from_sp(self[salkey], self[preskey],
                                [self.coordinates.x for _ in self[salkey]],
                                [self.coordinates.y for _ in self[salkey]])
            CT = gsw.ct_from_t(SA, self[tempkey], self[preskey])
            rho = gsw.rho(SA, CT, self[preskey])
            return self._addkeydata(rhokey, np.asarray(rho))
        else:
            raise FieldError("salinity, temperature, and pressure required")

    def add_depth(self, preskey="pressure", rhokey="density", depthkey="depth"):
        """ Use density and pressure to calculate depth.

        Args:

            preskey (str): data key to use for pressure
            rhokey (str): data key to use for in-situ density
            depthkey (str): data key to use for depth
        """
        if preskey not in self.fields:
            raise FieldError("add_depth requires a pressure field")
        if rhokey not in self.fields:
            raise FieldError("add_depth requires a density field")
        rho = self[rhokey].copy()

        # remove initial NaNs in Rho by replacing them with the first non-NaN
        idx = 0
        while np.isnan(rho.iloc[idx]):
            idx += 1
        rho.iloc[:idx] = rho.iloc[idx]

        dp = np.hstack([0.0, np.diff(self[preskey])])
        dz = dp / (rho.interpolate() * 9.81) * 1e4
        depth = np.cumsum(dz)
        return self._addkeydata(depthkey, depth)

    def add_Nsquared(self, rhokey="density", depthkey="depth", N2key="N2", s=0.2):
        """ Calculate the squared buoyancy frequency, based on in-situ density.
        Uses a smoothing spline to compute derivatives.

        Args:

            rhokey (str): data key to use for in-situ density
            depthkey (str): data key to use for depth
            N2key (str): data key to use for N^2
            s (float): spline smoothing factor (smaller values give a noisier result)
        """
        if rhokey not in self.fields:
            raise FieldError("in-situ density required")
        msk = self.nanmask((rhokey, depthkey))
        rho = self[rhokey][~msk]
        z = self[depthkey][~msk]
        rhospl = scipy.interpolate.UnivariateSpline(z, rho, s=s)
        drhodz = np.asarray([-rhospl.derivatives(_z)[1] for _z in z])
        N2 = np.empty(len(self), dtype=np.float64)
        N2[msk] = np.nan
        N2[~msk] = -9.81 / rho * drhodz
        return self._addkeydata(N2key, N2)

    def add_shear(self, depthkey="depth", ukey="u_velocity", vkey="v_velocity",
                  dudzkey="dudz", dvdzkey="dvdz", s=None):
        """ Compute the velocity shear for *u* and *v*. If *s* is not None,
        smooth the data with a gaussian filter before computing the derivative.

        Args:

            depthkey (str): data key to use for depth in meters
            vkey,ukey (str): data key to use for u,v velocity
            dudzkey (str): data key to use for u velocity shears
            dvdzkey (str): data key to use for v velocity shears
        """
        if ukey not in self.fields or vkey not in self.fields:
            raise FieldError("u and v velocity required")
        if s is not None:
            u = scipy.ndimage.filters.gaussian_filter1d(self[ukey], s)
            v = scipy.ndimage.filters.gaussian_filter1d(self[vkey], s)
        else:
            u = self[ukey]
            v = self[vkey]

        dudz = util.diff1(u.values, self[depthkey].values)
        dvdz = util.diff1(v.values, self[depthkey].values)
        self._addkeydata(dudzkey, dudz)
        self._addkeydata(dvdzkey, dvdz)
        return

def CTDCast(pres, sal, temp, **kw):
    """ Convenience function for creating CTD profiles. """
    kw["pressure"] = pres
    kw["salinity"] = sal
    kw["temperature"] = temp
    return Cast(**kw)

def XBTCast(depth, temp, **kw):
    """ Convenience function for creating XBT profiles. """
    kw["depth"] = depth
    kw["temperature"] = temp
    return Cast(**kw)

def LADCP(depth, uvel, vvel, **kw):
    """ Convenience function for creating LADCP profiles. """
    kw["depth"] = depth
    kw["u_velocity"] = uvel
    kw["v_velocity"] = vvel
    return Cast(**kw)

class CastCollection(NarwhalBase, collections.Sequence):
    """ A CastCollection is an indexable collection of Cast instances.

    Example:

        Create from casts or an iterable ordered sequence of casts::

            CastCollection(cast1, cast2, cast3...)

        or::

            CastCollection([cast1, cast2, cast3...])
    """
    _type = "castcollection"

    def __init__(self, *args):
        if len(args) == 0:
            self.casts = []
        elif isinstance(args[0], AbstractCast):
            self.casts = list(args)
        elif (len(args) == 1) and all(isinstance(a, AbstractCast) for a in args[0]):
            self.casts = args[0]
        else:
            raise TypeError("Arguments must be Casts or a collection of Casts")
        return

    def __len__(self):
        return len(self.casts)

    def __getitem__(self, key):
        if isinstance(key, (int, np.int32, np.int64)):
            return self.casts.__getitem__(key)
        elif isinstance(key, slice):
            return type(self)(self.casts.__getitem__(key))
        elif all(key in cast.data for cast in self.casts):
            return np.vstack([a[key] for a in self.casts]).T
        elif all(key in cast.properties for cast in self.casts):
            return [cast.properties[key] for cast in self.casts]
        else:
            raise KeyError("Key {0} not found in all casts".format(key))

    def __eq__(self, other):
        if not isinstance(self, type(other)):
            return False
        if len(self) != len(other):
            return False
        for (ca, cb) in zip(self, other):
            if ca != cb:
                return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def __contains__(self, cast):
        return True if (cast in self.casts) else False

    def __iter__(self):
        return (a for a in self.casts)

    def __add__(self, other):
        if isinstance(other, AbstractCastCollection):
            return CastCollection(list(a for a in itertools.chain(self.casts, other.casts)))
        elif isinstance(other, AbstractCast):
            return CastCollection(self.casts + [other])
        else:
            raise TypeError("no rule to add {0} to CastCollection".format(type(other)))

    def __repr__(self):
        s = "CastCollection with {n} casts:".format(n=len(self.casts))
        i = 0
        while i != 10 and i != len(self.casts):
            c = self.casts[i]
            lon = c.coordinates[0] or np.nan
            lat = c.coordinates[1] or np.nan
            s +=  ("\n  {num:3g} {typestr:6s} {lon:3.3f} {lat:2.3f}    "
                    "{keys}".format(typestr="Cast", num=i+1,
                                    lon=lon, lat=lat, keys=c.fields[:8]))
            if len(c.fields) > 8:
                s += " ..."
            i += 1
        if len(self.casts) > 10:
            s += "\n  (...)"
        return s

    @property
    def coordinates(self):
        return Multipoint([c.coordinates for c in self], crs=LonLatWGS84)

    def add_bathymetry(self, bathymetry):
        """ Reference Bathymetry instance `bathymetry` to CastCollection.

        Args:

            bathymetry (Bathymetry): bathymetric dataset to associate with
                casts
        """
        for cast in self.casts:
            if hasattr(cast, "coordinates"):
                cast.properties["depth"] = bathymetry.atpoint(cast.coordinates)
            else:
                cast.properties["depth"] = np.nan
                sys.stderr.write("bathymetry not added because cast location unknown\n")
        return

    def castwhere(self, key, value):
        """ Return the first cast where cast.properties[key] == value """
        for cast in self.casts:
            if cast.properties.get(key, None) == value:
                return cast
        raise LookupError("Cast not found with {0} = {1}".format(key, value))

    def castswhere(self, key, values=None):
        """ Return all casts satisfying criteria. Criteria are specified using
        one of the following patterns:

        Args:

            f (function): all casts satisfying `f(cast) == True` are returned

            k (str) and f (function): all casts for which `f(cast[key]) == True` are returned

            k (str) and L (iterable): all casts for which `cast[key] is in L ==
                True` are returned with a property key that is in
                `values` (iterable)
        """
        casts = []
        if values is None:
            if hasattr(key, "__call__"):
                return CastCollection([c for c in self if key(c)])
            else:
                raise NarwhalError("When one argument is given, it must be a function")
        if hasattr(values, "__call__"):
            func = values
            for cast in self.casts:
                if func(cast.properties[key]):
                    casts.append(cast)
        else:
            if not isinstance(values, collections.Container) or isinstance(values, str):
                values = (values,)
            for cast in self.casts:
                if cast.properties.get(key, None) in values:
                    casts.append(cast)
        return CastCollection(casts)

    def select(self, key, values):
        """ Return an CastCollection of Casts with selected where `key::str`
        equals `values::Iterable`
        """
        casts = [self.castwhere(key, v) for v in values]
        return CastCollection(casts)

    def nearest_to_point(self, point):
        """ Return the cast nearest to a karta Point, as well as the distance """
        distances = [point.distance(pt) for pt in self.coordinates]
        idx_min = np.argmin(distances)
        return self[idx_min], distances[idx_min]

    def defray(self):
        """ Pad casts to all have the same length, and return a copy.

        Warning: does not correct differing pressure bins, which require
        explicit interpolation.
        """
        n = max(len(c) for c in self)
        casts = []
        for cast_ in self:
            cast = copy.deepcopy(cast_)
            if len(cast) < n:
                dif = n - len(cast)
                cast.extend(dif)
                casts.append(cast)
            else:
                casts.append(cast)
        return CastCollection(casts)

    def asarray(self, key):
        """ Naively return values as an array, assuming that all casts are
        indexed with the same pressure levels.

        Args:

            key (str): property to return
        """
        nrows = max(len(cast) for cast in self.casts)
        arr = np.nan * np.empty((nrows, len(self.casts)), dtype=np.float64)
        for i, cast in enumerate(self.casts):
            arr[:len(cast), i] = cast[key]
        return arr

    def projdist(self):
        """ Return the cumulative distances from the cast to cast.
        """
        if (np.nan, np.nan) in (c.p["coordinates"] for c in self):
            raise AttributeError("all casts must contain non-NaN coordinates")
        cumulative = [0]
        prevcast = self.casts[0]
        for cast in self.casts[1:]:
            cumulative.append(cumulative[-1] + prevcast.coordinates.distance(cast.coordinates))
            prevcast = cast
        return np.asarray(cumulative, dtype=np.float64)

    def asdict(self):
        """ Return a representation of the Cast as a Python dictionary.
        """
        d = dict(__schemaversion__=2.0,
                 type="castcollection")
        d["casts"] = [cast.asdict() for cast in self.casts]
        return d

def load(fnm):
    """ Guess a file format based on filename extension and attempt to read it.
    """
    base, ext = os.path.splitext(fnm)
    if ext.lower() in (".h5", ".hdf"):
        return load_hdf(fnm)
    elif ext.lower() in (".nwl", ".nwz", ".json"):
        return load_json(fnm)
    else:
        raise NameError("File extension not recognized. "
                        "Try a format-specific read function instead.")

def load_hdf(fnm):
    """ Read HDF-formatted measurement data from `fnm::string`. """
    return fromdict(iohdf.read(fnm))

def load_json(fnm):
    """ Read JSON-formatted measurement data from `fnm::string`. """
    d = iojson.read(fnm)
    if d.get("__schemaversion__", 0.0) >= 2.0:
        return fromdict(d)
    else:
        # Try reading using schema version 1
        return iojson._fromjson_old(d, Cast, CastCollection)

def fromdict(d):
    """ Convert a dictionary to a Cast instance. """

    if "type" not in d:
        raise KeyError("dictionary missing `type` key")

    if d["type"] == "castcollection":
        return CastCollection([fromdict(cast) for cast in d["casts"]])

    elif d["type"] == "cast":
        data = d["data"]
        properties = d["properties"]
        for k,v in properties.items():
            if k.lower() in ("time", "timestamp", "date", "datetime"):
                try:
                    properties[k] = dateutil.parser.parse(v)
                except AttributeError:
                    # just read it as is
                    # this happens when value is a float
                    properties[k] = v

        data.update(properties)
        return Cast(**data)

    else:
        raise TypeError("'{0}' not a valid narwhal type".format(d["type"]))

def read_woce_netcdf(fnm):
    """ Read a CTD cast from a WOCE NetCDF file. """

    def getvariable(nc, key):
        return nc.variables[key].data.copy()

    nc = scipy.io.netcdf_file(fnm)
    coords = (getvariable(nc, "longitude")[0], getvariable(nc, "latitude")[0])

    pres = getvariable(nc, "pressure")
    sal = getvariable(nc, "salinity")
    salqc = getvariable(nc, "salinity_QC")
    sal[salqc!=2] = np.nan
    temp = getvariable(nc, "temperature")
    # tempqc = getvariable(nc, "temperature_QC")
    # temp[tempqc!=2] = np.nan
    oxy = getvariable(nc, "oxygen")
    oxyqc = getvariable(nc, "oxygen_QC")
    oxy[oxyqc!=2] = np.nan

    date = getvariable(nc, "woce_date")
    time = getvariable(nc, "woce_time")
    return CTDCast(pres, sal, temp, oxygen=oxy,
                   coordinates=coords,
                   properties={"woce_time":time, "woce_date":date})

class AbstractCast(six.with_metaclass(abc.ABCMeta)):
    pass

class AbstractCastCollection(six.with_metaclass(abc.ABCMeta)):
    pass

class NarwhalError(Exception):
    pass

class FieldError(TypeError):
    pass

AbstractCast.register(Cast)
AbstractCastCollection.register(CastCollection)
