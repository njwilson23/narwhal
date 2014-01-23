""" Module for handling the serialization of Cast- and CastCollection-like
objects to persistent files. """

import json
import copy
import datetime
import numpy

def castasdict(cast):
    scalars = [key for key in cast.data if not hasattr(cast[key], "__iter__")]
    vectors = [key for key in cast.data if hasattr(cast[key], "__iter__")]
    dscalar, dvector = {}, {}
    for key in scalars:
        if isinstance(cast[key], datetime.datetime):
            dscalar[key] = cast[key].strftime("%Y-%m-%d %H:%M:%S")
        else:
            dscalar[key] = cast[key]
    for key in vectors:
        if isinstance(cast[key], numpy.ndarray):
            dvector[key] = cast[key].tolist()
        else:
            dvector[key] = list(cast[key])
    d = dict(type="ctd_cast", scalars=dscalar, vectors=dvector, coords=cast.coords)
    return d

def dictascast(d, obj):
    """ Read a file-like stream and construct an object with a Cast-like
    interface. """
    d_ = copy.copy(d)
    _ = d_.pop("type")
    coords = d_.pop("coords")
    p = d_["vectors"].pop("pres")
    d_["vectors"].update(d_["scalars"])
    cast = obj(p, coords=coords, **d_["vectors"])
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
    d = dict(type="ctd_collection", casts=casts)
    json.dump(d, f)
    return

