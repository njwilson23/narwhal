# -*- coding: utf-8 -*-
"""
Cast and CastCollection classes for managing CTD observations
"""

import os
import sys
import abc
import collections
import itertools
import json
import gzip
import copy
import datetime, dateutil
from functools import reduce
import six
import numpy as np
import pandas
from scipy import ndimage
from scipy import sparse as sprs
from scipy.interpolate import UnivariateSpline
from scipy.io import netcdf_file
from karta import Point, Multipoint
from karta.crs import LonLatWGS84
from . import gsw
from . import units
from . import util
from . import iojson
from . import iohdf

# Global physical constants
G = 9.8
OMEGA = 2*np.pi / 86400.0

class Cast(object):
    """ A Cast is a set of referenced measurements associated with a single
    coordinate.

    Vector water properties and scalar metadata are provided as keyword
    arguments. There are several reserved keywords:

    coords::iterable[2]
        the geographic coordinates of the observation
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
        if self.coords is not None:
            coords = tuple(round(c, 3) for c in self.coords)
        else:
            coords = (None, None)
        s = "Cast (" + "".join([str(k)+", " for k in self.fields])
        # cut off the final comma
        s = s[:-2] + ") at {0}".format(coords)
        return s

    def __getitem__(self, key):
        if isinstance(key, int):
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

        elif isinstance(key, int):
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
    def coords(self):
        return self.properties["coordinates"]

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
        """ Add `n::int` NaN depth levels to cast. """
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

        y::string
            name of property to interpolate

        x::string
            name of reference property

        v::iterable
            vector of values for x

        force::bool
            whether to coerce x to be monotonic (default False)

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

    def save_json(self, fnm, binary=True):
        """ Save a JSON-formatted representation to a file at `fnm::string`.
        """
        if hasattr(fnm, "write"):
            iojson.write(fnm, self.asdict(), binary=binary)
        else:
            if binary:
                if os.path.splitext(fnm)[1] != ".nwz":
                    fnm = fnm + ".nwz"
                with gzip.open(fnm, "wb") as f:
                    iojson.write(f, self.asdict(), binary=True)
            else:
                if os.path.splitext(fnm)[1] != ".nwl":
                    fnm = fnm + ".nwl"
                with open(fnm, "w") as f:
                    iojson.write(f, self.asdict(), binary=False)
        return

    def save_hdf(self, fnm):
        return iohdf.write(fnm, self.asdict())

    def add_density(self, salkey="salinity", tempkey="temperature", preskey="pressure", rhokey="rho"):
        """ Add in-situ density computed from salinity, temperature, and
        pressure to fields. Return the field name.
        
        salkey::string
            data key to use for salinity

        tempkey::string
            data key to use for in-situ temperature

        preskey::string
            data key to use for pressure

        rhokey::string
            data key to use for in-situ density
        """
        if salkey in self.fields and tempkey in self.fields and preskey in self.fields:
            SA = gsw.sa_from_sp(self[salkey], self[preskey],
                                [self.coords[0] for _ in self[salkey]],
                                [self.coords[1] for _ in self[salkey]])
            CT = gsw.ct_from_t(SA, self[tempkey], self[preskey])
            rho = gsw.rho(SA, CT, self[preskey])
            return self._addkeydata(rhokey, np.asarray(rho))
        else:
            raise FieldError("salinity, temperature, and pressure required")

    def add_depth(self, preskey="pressure", rhokey="rho", depthkey="depth"):
        """ Use density and pressure to calculate depth.
        
        preskey::string
            data key to use for pressure

        rhokey::string
            data key to use for in-situ density

        depthkey::string
            data key to use for depth
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
        dz = dp / (rho.interpolate() * G) * 1e4
        depth = np.cumsum(dz)
        return self._addkeydata(depthkey, depth)

    def add_Nsquared(self, rhokey="rho", depthkey="z", N2key="N2", s=0.2):
        """ Calculate the squared buoyancy frequency, based on in-situ density.
        Uses a smoothing spline to compute derivatives.
        
        rhokey::string
            data key to use for in-situ density

        depthkey::string
            data key to use for depth

        N2key::string
            data key to use for N^2

        s::float
            spline smoothing factor (smaller values give a noisier result)
        """
        if rhokey not in self.fields:
            raise FieldError("in-situ density required")
        msk = self.nanmask((rhokey, depthkey))
        rho = self[rhokey][~msk]
        z = self[depthkey][~msk]
        rhospl = UnivariateSpline(z, rho, s=s)
        drhodz = np.asarray([-rhospl.derivatives(_z)[1] for _z in z])
        N2 = np.empty(len(self), dtype=np.float64)
        N2[msk] = np.nan
        N2[~msk] = -G / rho * drhodz
        return self._addkeydata(N2key, N2)

    def baroclinic_modes(self, nmodes, ztop=10, N2key="N2", depthkey="z"):
        """ Calculate the baroclinic normal modes based on linear
        quasigeostrophy and the vertical stratification. Return the first
        `nmodes::int` deformation radii and their associated eigenfunctions.

        **Parameters**

        ztop::float
            the depth at which to cut off the profile, to avoid surface effects

        N2key::string
            data key to use for N^2

        depthkey::string
            data key to use for depth
        """
        if N2key not in self.fields or depthkey not in self.fields:
            raise FieldError("buoyancy frequency and depth required")

        igood = ~self.nanmask((N2key, depthkey))
        N2 = self[N2key][igood]
        dep = self[depthkey][igood]

        itop = np.argwhere(dep > ztop)[0]
        N2 = N2[itop:].values
        dep = dep[itop:].values

        h = np.diff(dep)
        assert all(h[0] == h_ for h_ in h[1:])     # requires uniform gridding for now

        f = 2*OMEGA * np.sin(self.coords[1])
        F = f**2/N2
        F[0] = 0.0
        F[-1] = 0.0
        F = sprs.diags(F, 0)

        D1 = util.sparse_diffmat(len(N2), 1, h[0])
        D2 = util.sparse_diffmat(len(N2), 2, h[0])

        T = sprs.diags(D1 * F.diagonal(), 0)
        M = T*D1 + F*D2
        lamda, V = sprs.linalg.eigs(M.tocsc(), k=nmodes+1, sigma=1e-8)
        Ld = 1.0 / np.sqrt(np.abs(np.real(lamda[1:])))
        return Ld, V[:,1:]

    def water_fractions(self, sources, tracers=("sal", "temp")):
        """ Compute water mass fractions based on *n* (>= 2) conservative
        tracers.

        sources::[tuple, tuple, ...]
            List of *n+1* tuples specifying prototype water masses in terms of
            *tracers*. Each tuple must have length *n*.

        tracers::[string, string, ...]
            *n* Cast fields to use as tracers [default: ("sal", "temp")].
        """
        n = len(tracers)
        if n < 2:
            raise NarwhalError("Two or more prototype waters required")

        m = self.nvalid(tracers)
        I = sprs.eye(m)
        A_ = np.array([[src[i] for src in sources] for i in range(n)])
        A = np.vstack([A_, np.ones(n+1, dtype=np.float64)])
        As = sprs.kron(I, A, "csr")
        b = np.zeros((n+1)*m)
        msk = self.nanmask(tracers)
        for i in range(n):
            b[i::n+1] = self[tracers[i]][~msk]
        b[n::n+1] = 1.0             # lagrange multiplier

        frac = sprs.linalg.spsolve(As, b)
        chis = [np.empty(len(self)) * np.nan for i in range(n+1)]
        for i in range(n+1):
            chis[i][~msk] = frac[i::n+1]
        return chis

    def add_shear(self, depthkey="depth", ukey="u_velocity", vkey="v_velocity", 
                  dudzkey="dudz", dvdzkey="dvdz", s=None):
        """ Compute the velocity shear for *u* and *v*. If *s* is not None,
        smooth the data with a gaussian filter before computing the derivative.

        depthkey::string
            data key to use for depth in meters

        vkey,ukey::string
            data key to use for u,v velocity

        dudzkey,dvdzkey::string
            data key to use for u,v velocity shears
        """
        if ukey not in self.fields or vkey not in self.fields:
            raise FieldError("u and v velocity required")
        if s is not None:
            u = ndimage.filters.gaussian_filter1d(self[ukey], s)
            v = ndimage.filters.gaussian_filter1d(self[vkey], s)
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

class CastCollection(collections.Sequence):
    """ A CastCollection is an indexable collection of Cast instances.

    Create from casts or an iterable ordered sequence of casts:

        CastCollection(cast1, cast2, cast3...)

    or

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
        if isinstance(key, int):
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
            lon = c.coords[0] or np.nan
            lat = c.coords[1] or np.nan
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
    def coords(self):
        return Multipoint([c.coords for c in self], crs=LonLatWGS84)

    def add_bathymetry(self, bathymetry):
        """ Reference Bathymetry instance `bathymetry` to CastCollection.

        bathymetry::Bathymetry2d
            bathymetry instance
        """
        for cast in self.casts:
            if hasattr(cast, "coords"):
                cast.properties["depth"] = bathymetry.atxy(*cast.coords)
            else:
                cast.properties["tdepth"] = np.nan
                sys.stderr.write("Warning: cast has no coordinates")
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

        - f::function, in which case all casts satisfying `f(cast) == True` are
          returned

        - k::string and f::function, in which case all casts for which
          `f(cast[key]) == True` are returned

        - k::string and L::Container, in which case all casts for which
          `cast[key] is in L == True` are returned

        with a property key that is in `values::Container`
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

        key::string
            property to return
        """
        nrows = max(len(cast) for cast in self.casts)
        arr = np.nan * np.empty((nrows, len(self.casts)), dtype=np.float64)
        for i, cast in enumerate(self.casts):
            arr[:len(cast), i] = cast[key]
        return arr

    def projdist(self):
        """ Return the cumulative distances from the cast to cast.
        """
        cumulative = [0]
        a = Point(self.casts[0].coords, crs=LonLatWGS84)
        for cast in self.casts[1:]:
            b = Point(cast.coords, crs=LonLatWGS84)
            cumulative.append(cumulative[-1] + a.distance(b))
            a = b
        return np.asarray(cumulative, dtype=np.float64)

    def thermal_wind(self, tempkey="temp", salkey="sal", rhokey=None,
                     dudzkey="dudz", ukey="u", overwrite=False):
        """ Compute profile-orthagonal velocity shear using hydrostatic thermal
        wind. In-situ density is computed from temperature and salinity unless
        *rhokey* is provided.

        Adds a U field and a ∂U/∂z field to each cast in the collection. As a
        side-effect, if casts have no "depth" field, one is added and populated
        from temperature and salinity fields.

        **Parameters**

        tempkey::string
            key to use for temperature if *rhokey* is None

        salkey::string
            key to use for salinity if *rhokey* is None

        rhokey::string
            key to use for density, or None [default: None]

        dudzkey::string
            key to use for ∂U/∂z, subject to *overwrite*

        ukey::string
            key to use for U, subject to *overwrite*

        overwrite::bool
            whether to allow cast fields to be overwritten if False, then
            *ukey* and *dudzkey* are incremented until there is no clash
        """
        if rhokey is None:
            rhokeys = []
            for cast in self.casts:
                rhokeys.append(cast.add_density())
            if any(r != rhokeys[0] for r in rhokeys[1:]):
                raise NarwhalError("Tried to add density field, but got inconsistent keys")
            else:
                rhokey = rhokeys[0]

        rho = self.asarray(rhokey)
        (m, n) = rho.shape

        for cast in self:
            if "z" not in cast.data.keys():
                cast.add_depth()

        drho = util.diff2_dinterp(rho, self.projdist())
        sinphi = np.sin([c.coords[1]*np.pi/180.0 for c in self.casts])
        dudz = (G / rho * drho) / (2*OMEGA*sinphi)
        u = util.uintegrate(dudz, self.asarray("z"))

        for (ic,cast) in enumerate(self.casts):
            cast._addkeydata(dudzkey, dudz[:,ic], overwrite=overwrite)
            cast._addkeydata(ukey, u[:,ic], overwrite=overwrite)
        return

    def thermal_wind_inner(self, tempkey="temp", salkey="sal", rhokey=None,
                           dudzkey="dudz", ukey="u", bottomkey="depth",
                           overwrite=False):
        """ Alternative implementation that creates a new cast collection
        consistng of points between the observation casts.

        Compute profile-orthagonal velocity shear using hydrostatic thermal
        wind. In-situ density is computed from temperature and salinity unless
        *rhokey* is provided.

        Adds a U field and a ∂U/∂z field to each cast in the collection. As a
        side-effect, if casts have no "depth" field, one is added and populated
        from temperature and salinity fields.

        **Parameters**

        tempkey::string
            key to use for temperature if *rhokey* is None

        salkey::string
            key to use for salinity if *rhokey* is None

        rhokey::string
            key to use for density, or None [default: None]

        dudzkey::string
            key to use for ∂U/∂z, subject to *overwrite*

        ukey::string
            key to use for U, subject to *overwrite*

        overwrite::bool
            whether to allow cast fields to be overwritten if False, then
            *ukey* and *dudzkey* are incremented until there is no clash
        """
        if rhokey is None:
            rhokeys = []
            for cast in self.casts:
                rhokeys.append(cast.add_density())
            if any(r != rhokeys[0] for r in rhokeys[1:]):
                raise NarwhalError("Tried to add density field, but found inconsistent keys")
            else:
                rhokey = rhokeys[0]

        rho = self.asarray(rhokey)
        (m, n) = rho.shape

        def avgcolumns(a, b):
            avg = a if len(a[~np.isnan(a)]) > len(b[~np.isnan(b)]) else b
            return avg

        # Add casts in intermediate positions
        midcasts = []
        for i in range(len(self.casts)-1):
            c1 = self[i].coords
            c2 = self[i+1].coords
            cmid = (0.5*(c1[0]+c2[0]), 0.5*(c1[1]+c2[1]))
            p = avgcolumns(self[i]["pressure"], self[i+1]["pressure"])
            t = avgcolumns(self[i]["temperature"], self[i+1]["temperature"])
            s = avgcolumns(self[i]["salinity"], self[i+1]["salinity"])
            cast = CTDCast(p, s, t, coords=cmid)
            if "depth" not in cast.fields:
                cast.add_density()
            cast.add_depth()
            cast.properties[bottomkey] = 0.5 * (self[i].properties[bottomkey] +
                                                self[i+1].properties[bottomkey])
            midcasts.append(cast)

        coll = CastCollection(midcasts)
        drho = util.diff2_inner(rho, self.projdist())
        sinphi = np.sin([c.coords[1]*np.pi/180.0 for c in midcasts])
        rhoavg = 0.5 * (rho[:,:-1] + rho[:,1:])
        dudz = (G / rhoavg * drho) / (2*OMEGA*sinphi)
        u = util.uintegrate(dudz, coll.asarray("z"))

        for (ic,cast) in enumerate(coll):
            cast._addkeydata(dudzkey, dudz[:,ic], overwrite=overwrite)
            cast._addkeydata(ukey, u[:,ic], overwrite=overwrite)
        return coll


    def eofs(self, key="temperature", zkey="depth", n_eofs=None):
        """ Compute the EOFs and EOF structures for *key*. Returns a cast with
        the structure functions, an array of eigenvectors (EOFs), and an array
        of eigenvalues.

        Requires all casts to have the same depth-gridding.
        
        **Parameters**

        key::string
            key to use for computing EOFs

        n_eofs::int
            number of EOFs to return
        """
        assert all(zkey in c.fields for c in self)
        assert all(all(self[0].data.index == c.data.index) for c in self[1:])

        if n_eofs is None:
            n_eofs = len(self)

        arr = self.asarray(key)
        msk = reduce(lambda a,b:a|b, (c.nanmask(key) for c in self))
        arr = arr[~msk,:]
        arr -= arr.mean()

        _,sigma,V = np.linalg.svd(arr)
        lamb = sigma**2/(len(self)-1)
        eofts = util.eof_timeseries(arr, V)

        c0 = self[0]
        c = Cast(**{zkey: c0[zkey][~msk]})
        for i in range(n_eofs):
            c._addkeydata("_eof".join([key, str(i+1)]), eofts[:,i])
        return c, lamb[:n_eofs], V[:,:n_eofs]

    def asdict(self):
        """ Return a representation of the Cast as a Python dictionary.
        """
        d = dict(__schemaversion__=2.0,
                 type="castcollection")
        d["casts"] = [cast.asdict() for cast in self.casts]
        return d

    def save_json(self, fnm, binary=True):
        """ Save a JSON-formatted representation to a file.

        fnm::string
            file name to save to
        """
        if hasattr(fnm, "write"):
            return iojson.write(fnm, self.asdict(), binary=binary)
        else:
            if binary:
                if os.path.splitext(fnm)[1] != ".nwz":
                    fnm = fnm + ".nwz"
                with gzip.open(fnm, "wb") as f:
                    iojson.write(f, self.asdict(), binary=True)
            else:
                if os.path.splitext(fnm)[1] != ".nwl":
                    fnm = fnm + ".nwl"
                with open(fnm, "w") as f:
                    iojson.write(f, self.asdict(), binary=False)
        return

    def save_hdf(self, fnm):
        return iohdf.write(fnm, self.asdict())

def load(fnm):
    """ Guess a file format based on filename extension and attempt to read it. 
    """
    base, ext = os.path.splitext(fnm)
    if ext.lower() == ".hdf":
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
        try:
            with open(fnm, "r") as f:
                d = json.load(f)
        except (UnicodeDecodeError,ValueError):
            with gzip.open(fnm, "rb") as f:
                s = f.read().decode("utf-8")
                d = json.loads(s)
        return _fromjson(d)

# Dictionary schema:
#
# { type        ->  str: *type*,
#   data        ->  { [key]?        ->  *value*,
#                     [key]?        ->  *value* }
#   properties  ->  { coordinates   ->  (float: *lon*, float: *lat*),
#                     date|time?    ->  str: datetime (ISO formatted),
#                     [key]?        ->  *value*,
#                     [key]?        ->  *value* }
#

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
                properties[k] = dateutil.parser.parse(v)

        data.update(properties)
        return Cast(**data)

    else:
        raise TypeError("'{0}' not a valid narwhal type".format(d["type"]))

def read_woce_netcdf(fnm):
    """ Read a CTD cast from a WOCE NetCDF file. """

    def getvariable(nc, key):
        return nc.variables[key].data.copy()

    nc = netcdf_file(fnm)
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
                   coords=coords,
                   properties={"woce_time":time, "woce_date":date})

##### FUNCTIONS FOR READING THE DEPRECATED JSON SCHEMA #####

def _fromjson(d):
    """ (DEPRECATED) Lower level function to (possibly recursively) convert
    JSON into narwhal object. This reads the older JSON schema.
    """

    typ = d.get("type", None)
    if typ == "cast":
        return dictascast(d, Cast)
    elif typ == "ctdcast":
        return dictascast(d, CTDCast)
    elif typ == "xbtcast":
        return dictascast(d, XBTCast)
    elif typ == "ladcpcast":
        return dictascast(d, LADCP)
    elif typ == "castcollection":
        casts = [_fromjson(castdict) for castdict in d["casts"]]
        return CastCollection(casts)
    elif typ is None:
        raise NarwhalError("couldn't read data type - file may be corrupt")
    else:
        raise LookupError("Invalid type: {0}".format(typ))

def findunit(unitname):
    """ (DEPRECATED - USED FOR _fromjson) """
    for name in units.__dict__:
        if str(units.__dict__[name]) == unitname:
            return units.__dict__[name]
    raise NameError("'{0}' not recognized as a unit".format(unitname))

def dictascast(d, obj):
    """ (DEPRECATED - USED FOR _fromjson)

    Read a file-like stream and construct an object with a Cast-like
    interface. """
    d_ = d.copy()
    d_.pop("type")
    coords = d_["scalars"].pop("coordinates")
    prop = d["scalars"]
    for (key, value) in prop.items():
        if "date" in key or "time" in key and isinstance(prop[key], str):
            try:
                prop[key] = dateutil.parser.parse(value)
            except (TypeError, ValueError):
                pass
    prop.update(d_["vectors"])
    cast = obj(coords=coords, **prop)
    return cast

############################################################

class AbstractCast(six.with_metaclass(abc.ABCMeta)):
    pass

class AbstractCastCollection(six.with_metaclass(abc.ABCMeta)):
    pass

class NarwhalError(Exception):
    pass

class FieldError(TypeError):
    pass

AbstractCast.register(Cast)
# AbstractCast.register(CTDCast)
# AbstractCast.register(XBTCast)
# AbstractCast.register(LADCP)
AbstractCastCollection.register(CastCollection)

