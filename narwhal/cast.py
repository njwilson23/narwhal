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
from functools import reduce
import six
import numpy as np
import pandas
from scipy import ndimage
from scipy import sparse as sprs
from scipy.interpolate import UnivariateSpline
from scipy.io import netcdf_file
from karta import Point, Multipoint
from . import units
from . import fileio
from . import gsw
from . import util

try:
    from karta.crs import crsreg
except ImportError:
    import karta as crsreg
LONLAT = crsreg.LONLAT
CARTESIAN = crsreg.CARTESIAN

# Global physical constants
G = 9.8
OMEGA = 2*np.pi / 86400.0


class Cast(object):
    """ A Cast is a set of referenced measurements associated with a single
    coordinate.

    Vector water properties and scalar metadata are provided as keyword
    arguments. There are several reserved keywords:

    coords::iterable[2]     the geographic coordinates of the observation
    zunit::Unit             the independent vector units [default: meter]
    zname::string           name for the independent vector [default: "z"]
    """

    _type = "cast"

    def __init__(self, z, coords=(None, None), zunits=units.meter, zname="z", **kwargs):

        self.properties = {}
        data = {}

        self.zunits = zunits
        self.zname = zname
        self.p = self.properties

        # Python 3 workaround
        try:
            items = kwargs.iteritems()
        except AttributeError:
            items = kwargs.items()

        # Populate vector and scalar data fields
        data[zname] = pandas.Series(data=z, name=zname)
        for (kw, val) in items:
            if isinstance(val, collections.Container) and \
                    not isinstance(val, str) and \
                    len(val) == len(z):
                data[kw] = pandas.Series(data=val, name=kw)
            else:
                self.properties[kw] = val
        self.properties["coordinates"] = tuple(coords)
        self.data = pandas.DataFrame(data)
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
                raise IndexError("{0} not within cast length "
                                 "({1})".format(key, len(self)))
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
                raise TypeError("Fields must be set from iterables with length equal to the cast")

        elif isinstance(key, int):
            raise KeyError("Cast object profiles are not mutable")
        else:
            raise KeyError("Cannot use {0} as a hash".format(key))
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
            empty_df = pandas.DataFrame({self.zname: np.nan * np.empty(n)})
            self.data = pandas.concat([self.data, empty_df], ignore_index=True)
        else:
            raise ValueError("Cast must be extended with 1 or more rows")
        return

    def interpolate(self, y, x, v, force=False):
        """ Interpolate property y as a function of property x at values given
        by vector x=v.

        y::string       name of property to interpolate
        x::string       name of reference property
        v::iterable     vector of values for x
        force::bool     whether to coerce x to be monotonic (defualt False)

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
            raise ValueError("x is not monotonic")

    def regrid(self, levels):
        """ Re-interpolate Cast at specified grid levels. Returns a new Cast. """
        # some low level voodoo
        ret = copy.deepcopy(self)
        newdata = pandas.DataFrame(index=levels)
        for key in self.fields:
            newdata[key] = np.interp(levels, self.data[self.zname], self[key],
                                     left=np.nan, right=np.nan)
        ret.data = newdata
        return ret

    def save(self, fnm, binary=True):
        """ Save a JSON-formatted representation to a file at `fnm::string`.
        """
        if hasattr(fnm, "write"):
            fileio.writecast(fnm, self, binary=binary)
        else:
            if binary:
                if os.path.splitext(fnm)[1] != ".nwz":
                    fnm = fnm + ".nwz"
                with gzip.open(fnm, "wb") as f:
                    fileio.writecast(f, self, binary=True)
            else:
                if os.path.splitext(fnm)[1] != ".nwl":
                    fnm = fnm + ".nwl"
                with open(fnm, "w") as f:
                    fileio.writecast(f, self, binary=False)
        return

    def add_density(self, salkey="sal", tempkey="temp", preskey="pres", rhokey="rho"):
        """ Add in-situ density computed from salinity, temperature, and
        pressure to fields. Return the field name.
        
        salkey::string              Data key to use for salinity
        tempkey::string             Data key to use for in-situ temperature
        preskey::string             Data key to use for pressure
        rhokey::string              Data key to use for in-situ density
        """
        if salkey in self.fields and tempkey in self.fields and \
                (self.zunits == units.decibar or preskey != "z"):
            SA = gsw.sa_from_sp(self[salkey], self[preskey],
                                [self.coords[0] for _ in self[salkey]],
                                [self.coords[1] for _ in self[salkey]])
            CT = gsw.ct_from_t(SA, self[tempkey], self[preskey])
            rho = gsw.rho(SA, CT, self[preskey])
            return self._addkeydata(rhokey, np.asarray(rho))
        else:
            raise FieldError("add_density requires salinity, temperature, and "
                             "pressure fields")

    def add_depth(self, preskey="pres", rhokey="rho", depthkey="z"):
        """ Use density and pressure to calculate depth.
        
        preskey::string             Data key to use for pressure
        rhokey::string              Data key to use for in-situ density
        depthkey::string            Data key to use for depth
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
        
        rhokey::string              Data key to use for in-situ density
        depthkey::string            Data key to use for depth
        N2key::string               Data key to use for N^2
        s::float                    Spline smoothing factor (smaller values
                                    give a noisier result)
        """
        if rhokey not in self.fields:
            raise FieldError("add_Nsquared requires in-situ density")
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

        Additional arguments
        --------------------

        ztop                        the depth at which to cut off the profile,
                                    to avoid surface effects
        N2key::string               Data key to use for N^2
        depthkey::string            Data key to use for depth
        """
        if N2key not in self.fields or depthkey not in self.fields:
            raise FieldError("baroclinic_modes requires buoyancy frequency and depth")

        igood = ~self.nanmask((N2key, depthkey))
        N2 = self[N2key][igood]
        dep = self[depthkey][igood]

        itop = np.argwhere(dep > ztop)[0]
        N2 = N2[itop:]
        dep = dep[itop:]

        h = np.diff(dep)
        assert all(h == h_ for h_ in h[1:])     # requires uniform gridding for now

        f = 4*OMEGA * math.sin(self.coords[1])
        F = f**2/N2
        F[0] = 0.0
        F[-1] = 0.0
        F = sprs.diags(F, 0)

        D1 = util.sparse_diffmat(len(self), 1, h)
        D2 = util.sparse_diffmat(len(self), 2, h)

        T = sparse.diags(D1 * F.diagonal(), 0)
        M = T*D1 + F*D2
        lamda, V = sprs.linalg.eigs(M.tocsc(), k=nmodes+1, sigma=1e-8)
        Ld = 1.0 / np.sqrt(np.abs(np.real(lamda[1:])))
        return Ld, V[:,1:]

    def water_fractions(self, sources, tracers=("sal", "temp")):
        """ Compute water mass fractions based on conservative tracers.
        `sources::[tuple, tuple, ...]` is a list of tuples giving the prototype water
        masses.

        tracers::[string, string]       must be fields in the CTDCast to use as
                                        conservative tracers
                                        [default: ("sal", "temp")].
        """

        if len(sources) != 3:
            raise ValueError("Three potential source waters must be given "
                             "(not {0})".format(len(sources)))
        n = self.nvalid(tracers)
        I = sprs.eye(n)
        A_ = np.array([[sources[0][0], sources[1][0], sources[2][0]],
                       [sources[0][1], sources[1][1], sources[2][1]],
                       [         1.0,          1.0,          1.0]])
        As = sprs.kron(I, A_, "csr")
        b = np.empty(3*n)
        msk = self.nanmask(tracers)
        b[::3] = self[tracers[0]][~msk]
        b[1::3] = self[tracers[1]][~msk]
        b[2::3] = 1.0               # lagrange multiplier

        frac = sprs.linalg.spsolve(As, b)
        mass1 = np.empty(len(self)) * np.nan
        mass2 = np.empty(len(self)) * np.nan
        mass3 = np.empty(len(self)) * np.nan
        mass1[~msk] = frac[::3]
        mass2[~msk] = frac[1::3]
        mass3[~msk] = frac[2::3]
        return (mass1, mass2, mass3)

    def add_shear(self, depthkey="z", ukey="u", vkey="v", dudzkey="dudz", dvdzkey="dvdz",
                  s=None):
        """ Compute the velocity shear for *u* and *v*. If *s* is not None,
        smooth the data with a gaussian filter before computing the derivative.

        depthkey::string            Data key to use for depth
        vkey,ukey::string           Data key to use for u,v velocity
        dudzkey,dvdzkey::string     Data key to use for u,v velocity shears
        """
        if ukey not in self.fields or vkey not in self.fields:
            raise FieldError("add_shear requires u and v velocity components")
        if depthkey == "z" and self.zunits != units.meter:
            raise FieldError("add_shear requires depth in meters")
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

def CTDCast(pres, sal, temp, coords=(None, None), **kw):
    """ Convenience function for creating CTD profiles. """
    kw["sal"] = sal
    kw["temp"] = temp
    return Cast(pres, zunits=units.decibar, zname="pres", coords=coords, **kw)

def XBTCast(depth, temp, coords=(None, None), **kw):
    """ Convenience function for creating XBT profiles. """
    kw["temp"] = temp
    return Cast(depth, zunits=units.meter, coords=coords, **kw)

def LADCP(depth, uvel, vvel, coords=(None, None), **kw):
    """ Convenience function for creating LADCP profiles. """
    kw["u"] = uvel
    kw["v"] = vvel
    return Cast(depth, zunits=units.meter, coords=coords, **kw)

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
            raise TypeError("Can only add castcollection and *cast types to "
                            "CastCollection")

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
        return Multipoint([c.coords for c in self], crs=LONLAT)

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
                raise ValueError("If one argument is given, it must be a function")
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

        key::string                     property to return
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
        a = Point(self.casts[0].coords, crs=LONLAT)
        for cast in self.casts[1:]:
            b = Point(cast.coords, crs=LONLAT)
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

        Parameters
        ----------

        tempkey::string     key to use for temperature if *rhokey* is None
        salkey::string      key to use for salinity if *rhokey* is None
        rhokey::string      key to use for density, or None [default: None]
        dudzkey::string     key to use for ∂U/∂z, subject to *overwrite*
        ukey::string        key to use for U, subject to *overwrite*
        overwrite::bool     whether to allow cast fields to be overwritten
                            if False, then *ukey* and *dudzkey* are incremented
                            until there is no clash
        """
        if rhokey is None:
            rhokeys = []
            for cast in self.casts:
                rhokeys.append(cast.add_density())
            if any(r != rhokeys[0] for r in rhokeys[1:]):
                raise NameError("Tried to add density field, but ended up with "
                                "different keys - aborting")
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

        Parameters
        ----------

        tempkey::string     key to use for temperature if *rhokey* is None
        salkey::string      key to use for salinity if *rhokey* is None
        rhokey::string      key to use for density, or None [default: None]
        dudzkey::string     key to use for ∂U/∂z, subject to *overwrite*
        ukey::string        key to use for U, subject to *overwrite*
        overwrite::bool     whether to allow cast fields to be overwritten
                            if False, then *ukey* and *dudzkey* are incremented
                            until there is no clash
        """
        if rhokey is None:
            rhokeys = []
            for cast in self.casts:
                rhokeys.append(cast.add_density())
            if any(r != rhokeys[0] for r in rhokeys[1:]):
                raise NameError("Tried to add density field, but ended up with "
                                "different keys - aborting")
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
            p = avgcolumns(self[i]["pres"], self[i+1]["pres"])
            t = avgcolumns(self[i]["temp"], self[i+1]["temp"])
            s = avgcolumns(self[i]["sal"], self[i+1]["sal"])
            cast = CTDCast(p, temp=t, sal=s, zname="pres", coords=cmid)
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

    def save(self, fnm, binary=True):
        """ Save a JSON-formatted representation to a file.

        fnm::string     File name to save to
        """
        if hasattr(fnm, "write"):
            fileio.writecastcollection(fnm, self, binary=binary)
        else:
            if binary:
                if os.path.splitext(fnm)[1] != ".nwz":
                    fnm = fnm + ".nwz"
                with gzip.open(fnm, "wb") as f:
                    fileio.writecastcollection(f, self, binary=True)
            else:
                if os.path.splitext(fnm)[1] != ".nwl":
                    fnm = fnm + ".nwl"
                with open(fnm, "w") as f:
                    fileio.writecastcollection(f, self, binary=False)
        return


def read(fnm):
    """ Convenience function for reading JSON-formatted measurement data from
    `fnm::string`.
    """
    try:
        with open(fnm, "r") as f:
            d = json.load(f)
    except (UnicodeDecodeError,ValueError) as e:
        with gzip.open(fnm, "rb") as f:
            s = f.read().decode("utf-8")
            d = json.loads(s)
    return _fromjson(d)

def _fromjson(d):
    """ Lower level function to (possibly recursively) convert JSON into
    narwhal object. """
    typ = d.get("type", None)
    if typ == "cast":
        return fileio.dictascast(d, Cast)
    elif typ == "ctdcast":
        return fileio.dictascast(d, CTDCast)
    elif typ == "xbtcast":
        return fileio.dictascast(d, XBTCast)
    elif typ == "ladcpcast":
        return fileio.dictascast(d, LADCP)
    elif typ == "castcollection":
        casts = [_fromjson(castdict) for castdict in d["casts"]]
        return CastCollection(casts)
    elif typ is None:
        raise AttributeError("couldn't read data type - file may be corrupt")
    else:
        raise LookupError("Invalid type: {0}".format(typ))

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
    return narwhal.CTDCast(pres, sal, temp, oxygen=oxy,
                           coords=coords,
                           properties={"woce_time":time, "woce_date":date})

class AbstractCast(six.with_metaclass(abc.ABCMeta)):
    pass

class AbstractCastCollection(six.with_metaclass(abc.ABCMeta)):
    pass

class FieldError(TypeError):
    pass

AbstractCast.register(Cast)
# AbstractCast.register(CTDCast)
# AbstractCast.register(XBTCast)
# AbstractCast.register(LADCP)
AbstractCastCollection.register(CastCollection)

