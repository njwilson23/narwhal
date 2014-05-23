""" Module for handling the serialization of Cast- and CastCollection-like
objects to persistent files. """

import json
import copy
import datetime
import dateutil.parser
import numpy
from karta import Point, geojson

def castasdict(cast):
    scalars = [key for key in cast.properties]
    vectors = [key for key in cast.data]
    dscalar, dvector = {}, {}
    for key in scalars:
        if isinstance(cast.properties[key], datetime.datetime):
            dscalar[key] = cast.properties[key].isoformat(sep=" ")
        else:
            dscalar[key] = cast.properties[key]
    for key in vectors:
        if isinstance(cast[key], numpy.ndarray):
            dvector[key] = cast[key].tolist()
        else:
            dvector[key] = list(cast[key])
    d = dict(type=cast._type, scalars=dscalar, vectors=dvector,
            coords=cast.coords, primarykey=cast.primarykey)
    return d

def dictascast(d, obj):
    """ Read a file-like stream and construct an object with a Cast-like
    interface. """
    d_ = copy.copy(d)
    _ = d_.pop("type")
    coords = d_.pop("coords")
    primkey = d_.pop("primarykey", "pres")
    p = d_["vectors"].pop(primkey)
    prop = d["scalars"]
    for (key, value) in prop.items():
        if "date" in key or "time" in key and isinstance(prop[key], str):
            try:
                prop[key] = dateutil.parser.parse(value)
            except (TypeError, ValueError):
                pass
    cast = obj(p, coords=coords, primarykey=primkey, properties=prop,
            **d_["vectors"])
    return cast

def dictascastcollection(d, castobj):
    """ Read a file-like stream and return a list of Cast-like objects.
    """
    casts = [dictascast(cast, castobj) for cast in d["casts"]]
    return casts

def writecast(f, cast):
    """ Write Cast data to a file-like stream. """
    d = castasdict(cast)
    json.dump(d, f)
    return

def writecastcollection(f, cc):
    """ Write CastCollection to a file-like stream. """
    casts = [castasdict(cast) for cast in cc]
    d = dict(type="castcollection", casts=casts)
    json.dump(d, f)
    return

def castcollection_as_geojson(cc):
    castpoints = (Point(c.coords, properties={"id":i})
                  for i, c in enumerate(cc))
    geojsonstring = geojson.printFeatureCollection(castpoints)
    return geojsonstring

