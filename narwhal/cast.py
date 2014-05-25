# -*- coding: utf-8 -*-
"""
Cast and CastCollection classes for managing CTD observations
"""

import os
import sys
import itertools
import collections
import json
import numpy as np
from karta import Point, LONLAT
from . import fileio
from . import gsw


# Global physical constants
G = 9.8
OMEGA = 2*np.pi / 86400.0


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
        """ Add data under *key*. If *key* already exists, iterates over
        [key]_2, [key]_3... until an unused identifier is found. Returns the
        key finally used.

        Use case: for automatic addition of fields.
        """
        key_ = key
        if not overwrite:
            i = 2
            while key_ in self.data:
                key_ = key + "_" + str(i)
                i += 1
        self.data[key_] = data
        return key_

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
            return np.interp(v, force_monotonic(self[x]), self[y])
        else:
            raise ValueError("x is not monotonic")

    def save(self, fnm):
        """ Save a JSON-formatted representation to a file.

        fnm::string     File name to save to
        """
        if os.path.splitext(fnm)[1] != ".nwl":
            fnm = fnm + ".nwl"
        with open(fnm, "w") as f:
            fileio.writecast(f, self)
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
        constemp = (gsw.ct_from_t(sa, t, p) for (sa, t, p)
                        in zip(self["sal"], self["temp"], self["pres"]))
        rho = np.array([gsw.rho(sa, ct, p) for (sa, ct, p)
                        in zip(self["sal"], constemp, self["pres"])])
        return self._addkeydata("rho", rho)

    def add_depth(self, rhokey=None):
        """ Use temperature, salinity, and pressure to calculate depth. If
        in-situ density is already in a field, *rhokey* can be provided to
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


class LADCP(Cast):
    """ Specialization of Cast for LADCP data. Requires *u* and *v* fields. """
    _type = "ladcpcast"

    def __init__(self, z, u, v, coords=None, properties=None,
                 primarykey="z", **kwargs):
        super(LADCP, self).__init__(z, u=u, v=v, coords=coords,
                                    properties=properties, primarykey=primarykey,
                                    **kwargs)
        return


class XBTCast(Cast):
    """ Specialization of Cast with temperature field. """
    _type = "xbtcast"

    def __init__(self, p, temp, coords=None, properties=None, **kwargs):
        super(XBTCast, self).__init__(p, temp=temp, coords=coords,
                                      properties=properties, **kwargs)
        return


class CastCollection(collections.Sequence):
    """ A CastCollection is an indexable collection of Cast instances """
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
                     dudzkey="dudz", ukey="u", overwrite=False):
        """ Compute profile-orthagonal velocity shear using hydrostatic thermal
        wind. In-situ density is computed from temperature and salinity unless
        *rhokey* is provided.

        Adds a U field and a ∂U/∂z field to each cast in the collection. As a
        side-effect, if casts have no "depth" field, one is added and populated
        from temperature and salinity fields.
        
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

        for cast in self:
            if "depth" not in cast.data.keys():
                cast.add_depth()

        drho = diff2(rho, self.projdist())
        sinphi = np.sin([c.coords[1]*np.pi/180.0 for c in self.casts])
        dudz = (G / rho * drho) / (2*OMEGA*sinphi)
        u = uintegrate(dudz, self.asarray("depth"))

        for (ic,cast) in enumerate(self.casts):
            cast._addkeydata(dudzkey, dudz[:,ic], overwrite=overwrite)
            cast._addkeydata(ukey, u[:,ic], overwrite=overwrite)
        return

    def thermal_wind_inner(self, tempkey="temp", salkey="sal", rhokey=None,
                           dudzkey="dudz", ukey="u", overwrite=False):
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

        for cast in self:
            if "depth" not in cast.data.keys():
                cast.add_depth()

        # Add casts in intermediate positions
        midcasts = []
        for i in range(len(self.casts)-1):
            c1 = self[i].coords
            c2 = self[i+1].coords
            cmid = (0.5*(c1[0]+c2[0]), 0.5*(c1[1]+c2[1]))
            z1 = self[i]["depth"]
            z2 = self[i+1]["depth"]
            z = z1 if len(z1[~np.isnan(z1)]) > len(~z2[np.isnan(z2)]) else z2
            midcasts[i] = Cast(z, primarykey="depth", coords=cmid)

        drho = diff2_inner(rho, self.projdist())
        sinphi = np.sin([c.coords[1]*np.pi/180.0 for c in midcasts])
        dudz = (G / rho * drho) / (2*OMEGA*sinphi)
        u = uintegrate(dudz, self.asarray("depth"))

        for (ic,cast) in enumerate(midcasts):
            cast._addkeydata(dudzkey, dudz[:,ic], overwrite=overwrite)
            cast._addkeydata(ukey, u[:,ic], overwrite=overwrite)
        return midcasts

    def save(self, fnm):
        """ Save a JSON-formatted representation to a file.

        fnm::string     File name to save to
        """
        if os.path.splitext(fnm)[1] != ".nwl":
            fnm = fnm + ".nwl"
        with open(fnm, "w") as f:
            fileio.writecastcollection(f, self)


def read(fnm):
    """ Convenience function for reading JSON-formatted measurement data. """
    with open(fnm, "r") as f:
        d = json.load(f)
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

