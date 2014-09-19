""" Module for handling the serialization of Cast- and CastCollection-like
objects to persistent files. """

import six
import json
import copy
import datetime
import dateutil.parser
import numpy
import pandas
from karta import Point, geojson
from . import units

def castasdict(cast):
    scalars = [key for key in cast.properties]
    vectors = list(cast.data.keys())
    dscalar, dvector = {}, {}
    for key in scalars:
        if isinstance(cast.properties[key], datetime.datetime):
            dscalar[key] = cast.properties[key].isoformat(sep=" ")
        else:
            dscalar[key] = cast.properties[key]
    for key in vectors:
        if isinstance(cast[key], numpy.ndarray):
            dvector[key] = cast[key].tolist()
        elif isinstance(cast[key], pandas.Series):
            dvector[key] = cast[key].values.tolist()
        else:
            dvector[key] = list(cast[key])
    d = dict(type=cast._type, scalars=dscalar, vectors=dvector,
            coords=cast.coords, zunits=str(cast.zunits), zname=str(cast.zname))
    return d

def findunit(unitname):
    for name in units.__dict__:
        if str(units.__dict__[name]) == unitname:
            return units.__dict__[name]
    raise NameError("'{0}' not recognized as a unit".format(unitname))

def dictascast(d, obj):
    """ Read a file-like stream and construct an object with a Cast-like
    interface. """
    d_ = d.copy()
    _ = d_.pop("type")
    coords = d_["scalars"].pop("coordinates")
    zunits = findunit(d_.pop("zunits", "meter"))
    zname = d_.pop("zname", "z")
    z = d_["vectors"].pop(zname)
    prop = d["scalars"]
    for (key, value) in prop.items():
        if "date" in key or "time" in key and isinstance(prop[key], str):
            try:
                prop[key] = dateutil.parser.parse(value)
            except (TypeError, ValueError):
                pass
    prop.update(d_["vectors"])
    cast = obj(z, coords=coords, zunits=zunits, zname=zname, **prop)
    return cast

def dictascastcollection(d, castobj):
    """ Read a file-like stream and return a list of Cast-like objects.
    """
    casts = [dictascast(cast, castobj) for cast in d["casts"]]
    return casts

def writecast(f, cast, binary=True):
    """ Write Cast data to a file-like stream. """
    d = castasdict(cast)
    if binary:
        s = json.dumps(d, indent=2)
        # f.write(bytes(s, "utf-8"))
        f.write(six.b(s))
    else:
        json.dump(d, f, indent=2)
    return

def writecastcollection(f, cc, binary=True):
    """ Write CastCollection to a file-like stream. """
    casts = [castasdict(cast) for cast in cc]
    d = dict(type="castcollection", casts=casts)
    if binary:
        s = json.dumps(d, indent=2)
        # f.write(bytes(s, "utf-8"))
        f.write(six.b(s))
    else:
        json.dump(d, f, indent=2)
    return

def castcollection_as_geojson(cc):
    castpoints = (Point(c.coords, properties={"id":i})
                  for i, c in enumerate(cc))
    geojsonstring = geojson.printFeatureCollection(castpoints)
    return geojsonstring

