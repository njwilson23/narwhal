# -*- coding: utf-8 -*-
"""
Cast and CastCollection classes for managing CTD observations
"""

import os
import sys
import itertools
import collections
import json
import gzip
from functools import reduce
import numpy as np
from scipy import ndimage
from scipy import sparse as sprs
from karta import Point, LONLAT
from . import fileio
from . import gsw
from . import util


# Global physical constants
G = 9.8
OMEGA = 2*np.pi / 86400.0


class Cast(object):
    """ A Cast is a set of referenced measurements associated with a single
    coordinate.
    
    Vector water properties are provided as keyword arguments. There are
    several reserved keywords:

    coords::iterable[2]     the geographic coordinates of the observation

    properties::dict        scalar metadata

    primarykey::string      the name of vertical measure. Usually pressure
                            ("pres"), but could be e.g. depth ("z")
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
        self.data = dict()
        self.data[primarykey] = np.asarray(p)

        # Python 3 workaround
        try:
            items = kwargs.iteritems()
        except AttributeError:
            items = kwargs.items()

        # Populate vector and scalar data fields
        for (kw, val) in items:
            if val is None:
                self.data[kw] = np.nan * np.empty(len(p))
            elif hasattr(val, "__len__") and len(val) == len(p):
                self.data[kw] = np.asarray(val)
            else:
                self.properties[kw] = val

        self._len = len(p)
        self._fields = [primarykey] + [a for a in kwargs]
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
            if key not in self._fields:
                self._fields.append(key)
        elif isinstance(key, int):
            raise KeyError("Cast object profiles are not mutable")
        else:
            raise KeyError("Cannot use {0} as a hash".format(key))
        return

    def __add__(self, other):
        if hasattr(other, "_type") and (other._type[-4:] == "cast"):
            return CastCollection(self, other)
        elif hasattr(other, "_type") and (other._type == "castcollection"):
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
            while key_ in self.data:
                key_ = key + "_" + str(i)
                i += 1
        if key_ not in self._fields:
            self._fields.append(key_)
        self.data[key_] = data
        return key_

    def nanmask(self, fields=None):
        """ Return a mask for observations containing at least one NaN. """
        if fields is None:
            fields = self._fields
        vectors = [v for (k,v) in self.data.items() if k in fields]
        return np.isnan(np.vstack(vectors).sum(axis=0))

    def nvalid(self):
        """ Return the number of complete (non-NaN) observations. """
        vectors = [a for a in self.data.values() if hasattr(a, "__iter__")]
        nv = sum(reduce(lambda a,b: (~np.isnan(a))&(~np.isnan(b)), vectors))
        return nv

    def interpolate(self, y, x, v, force=False):
        """ Interpolate property y as a function of property x at values given
        by vector x=v.

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
            return np.interp(v, util.force_monotonic(self[x]), self[y])
        else:
            raise ValueError("x is not monotonic")

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


class CTDCast(Cast):
    """ Specialization of Cast guaranteed to have salinity and temperature
    fields. """
    _type = "ctdcast"

    def __init__(self, p, sal, temp, coords=None, properties=None,
                 **kwargs):
        super(CTDCast, self).__init__(p, sal=sal, temp=temp, coords=coords,
                                      properties=properties, **kwargs)
        return

    def add_density(self):
        """ Add in-situ density to fields, and return the field name. """
        SA = [gsw.sa_from_sp(sp, p, self.coords[0], self.coords[1])
                    for (sp, p) in zip(self["sal"], self["pres"])]
        CT = (gsw.ct_from_t(sa, t, p) for (sa, t, p) in zip(SA, self["temp"], self["pres"]))
        rho = [gsw.rho(sa, ct, p) for (sa, ct, p) in zip(SA, CT, self["pres"])]
        return self._addkeydata("rho", np.asarray(rho))

    def add_depth(self, rhokey=None):
        """ Use temperature, salinity, and pressure to calculate depth. If
        in-situ density is already in a field, `rhokey::string` can be provided to
        avoid recalculating it. """
        if rhokey is None:
            rhokey = self.add_density()
        rho = self[rhokey]

        # remove initial NaNs by replacing them with the first non-NaN
        nnans = 0
        r = rho[0]
        while np.isnan(r):
            nnans += 1
            r = rho[nnans]
        rho[:nnans] = rho[nnans+1]

        dp = np.hstack([self["pres"][0], np.diff(self["pres"])])
        dz = dp / (rho * G) * 1e4
        depth = np.cumsum(dz)
        return self._addkeydata("depth", depth)

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
        n = self.nvalid()
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


class LADCP(Cast):
    """ Specialization of Cast for LADCP data. Requires *u* and *v* fields. """
    _type = "ladcpcast"

    def __init__(self, z, u, v, coords=None, properties=None,
                 primarykey="z", **kwargs):
        super(LADCP, self).__init__(z, u=u, v=v, coords=coords,
                                    properties=properties, primarykey=primarykey,
                                    **kwargs)
        return

    def add_shear(self, sigma=None):
        """ Compute the velocity shear for *u* and *v*. If *sigma* is not None,
        smooth the data with a gaussian filter before computing the derivative.
        """
        if sigma is not None:
            u = ndimage.filters.gaussian_filter1d(self["u"], sigma)
            v = ndimage.filters.gaussian_filter1d(self["v"], sigma)
        else:
            u = self["u"]
            v = self["v"]

        dudz = util.diff1(u, self["z"])
        dvdz = util.diff1(v, self["z"])
        self._addkeydata("dudz", dudz)
        self._addkeydata("dvdz", dvdz)
        return


class XBTCast(Cast):
    """ Specialization of Cast with temperature field. """
    _type = "xbtcast"

    def __init__(self, p, temp, coords=None, properties=None, **kwargs):
        super(XBTCast, self).__init__(p, temp=temp, coords=coords,
                                      properties=properties, **kwargs)
        return


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
        elif isinstance(args[0], Cast):
            self.casts = list(args)
        elif (len(args) == 1) and \
                (False not in (isinstance(a, Cast) for a in args[0])):
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

    def __eq__(self, other):
        if not hasattr(other, "_type") or (self._type != other._type):
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
        if hasattr(other, "_type") and (other._type == "castcollection"):
            return CastCollection(list(a for a in itertools.chain(self.casts, other.casts)))
        elif hasattr(other, "_type") and (other._type[-4:] == "cast"):
            return CastCollection(self.casts + [other])
        else:
            raise TypeError("Can only add castcollection and *cast types to "
                            "CastCollection")

    def __repr__(self):
        s = "CastCollection with {n} casts:".format(n=len(self.casts))
        i = 0
        while i != 10 and i != len(self.casts):
            c = self[i]
            s +=  ("\n  {num:3g} {typestr:6s} {lon:3.3f} {lat:2.3f}    "
                    "{keys}".format(typestr=c._type[:-4], num=i+1,
                                    lon=c.coords[0], lat=c.coords[1],
                                    keys=c._fields[:8]))
            if len(c._fields) > 8:
                s += " ..."
            i += 1
        if len(self.casts) > 10:
            s += "\n  (...)"
        return s

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

    def castwhere(self, key, value):
        """ Return the first cast where cast.properties[key] == value """
        for cast in self.casts:
            if cast.properties.get(key, None) == value:
                return cast

    def castswhere(self, key, value):
        """ Return all casts where cast.properties[key] == value """
        casts = []
        for cast in self.casts:
            if cast.properties.get(key, None) == value:
                casts.append(cast)
        return casts

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
            if False in (r == rhokeys[0] for r in rhokeys[1:]):
                raise NameError("Tried to add density field, but ended up with "
                                "different keys - aborting")
            else:
                rhokey = rhokeys[0]

        rho = self.asarray(rhokey)
        (m, n) = rho.shape

        for cast in self:
            if "depth" not in cast.data.keys():
                cast.add_depth()

        drho = util.diff2_dinterp(rho, self.projdist())
        sinphi = np.sin([c.coords[1]*np.pi/180.0 for c in self.casts])
        dudz = (G / rho * drho) / (2*OMEGA*sinphi)
        u = util.uintegrate(dudz, self.asarray("depth"))

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
            if False in (r == rhokeys[0] for r in rhokeys[1:]):
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
            cast = CTDCast(p, temp=t, sal=s, primarykey="pres", coords=cmid)
            cast.add_depth()
            cast.properties[bottomkey] = 0.5 * (self[i].properties[bottomkey] +
                                                self[i+1].properties[bottomkey])
            midcasts.append(cast)

        coll = CastCollection(midcasts)
        drho = util.diff2_inner(rho, self.projdist())
        sinphi = np.sin([c.coords[1]*np.pi/180.0 for c in midcasts])
        rhoavg = 0.5 * (rho[:,:-1] + rho[:,1:])
        dudz = (G / rhoavg * drho) / (2*OMEGA*sinphi)
        u = util.uintegrate(dudz, coll.asarray("depth"))

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
            # s = gzip.decompress(sz).decode("utf-8")
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

