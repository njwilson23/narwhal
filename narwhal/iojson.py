""" Module for handling the serialization of Cast- and CastCollection-like
objects to persistent files. """

import json
import gzip
import dateutil
import six

# This class coerces numpy values into Python types for JSON serialization. 
class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, o):
        try:
            if o.dtype in ("int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"):
                return int(o)
            elif o.dtype in ("float16", "float32", "float64", "float128"):
                return float(o)
            elif o.dtype in ("complex64", "complex128", "complex256"):
                return complex(o)
            else:
                raise TypeError("not a recognized type: {0}".format(o.dtype))
        except (AttributeError, TypeError):
            return json.JSONEncoder.default(self, o)

def write_text(fnm, d):

    try:
        if hasattr(fnm, "write"):
            f = fnm
        else:
            f = open(fnm, "w")

        json.dump(d, f, indent=2, cls=NumpyJSONEncoder)

    finally:
        f.close()
    return

def write_binary(fnm, d):

    try:
        if hasattr(fnm, "write"):
            f = fnm
        else:
            f = gzip.open(fnm, "wb")

        jsonstring = json.dumps(d, indent=0, cls=NumpyJSONEncoder)
        f.write(six.b(jsonstring))

    finally:
        f.close()
    return

def read(fnm):
    """ Read JSON-formatted measurement data from `fnm::string`. """
    try:
        with open(fnm, "r") as f:
            d = json.load(f)
    except (UnicodeDecodeError,ValueError):
        with gzip.open(fnm, "rb") as f:
            s = f.read().decode("utf-8")
            d = json.loads(s)
    return d


##### FUNCTIONS FOR READING THE DEPRECATED JSON SCHEMA #####

def _fromjson_old(d, cast_constructor, collection_constructor):
    """ (DEPRECATED) Lower level function to (possibly recursively) convert
    JSON into narwhal object. This reads the older JSON schema (version < 2.0).
    """

    typ = d.get("type", None)
    if typ in ("cast", "ctdcast", "xbtcast", "ladcpcast"):
        return dictascast_old(d, cast_constructor)
    elif typ == "castcollection":
        casts = [_fromjson_old(castdict, 
                               cast_constructor,
                               collection_constructor)
                            for castdict in d["casts"]]
        return collection_constructor(casts)
    elif typ is None:
        raise IOError("couldn't read data type - file may be corrupt")
    else:
        raise LookupError("Invalid type: {0}".format(typ))

def dictascast_old(d, constructor):
    """ (DEPRECATED - USED FOR _fromjson_old)

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
    return constructor(coordinates=coords, **prop)

############################################################



#def writecast(f, cast, binary=True):
#    """ Write Cast data to a file-like stream. """
#    d = cast.asdict()
#    if binary:
#        s = json.dumps(d, indent=2)
#        # f.write(bytes(s, "utf-8"))
#        f.write(six.b(s))
#    else:
#        json.dump(d, f, indent=2)
#    return
#
#def writecastcollection(f, cc, binary=True):
#    """ Write CastCollection to a file-like stream. """
#    d = cc.asdict()
#    if binary:
#        s = json.dumps(d, indent=2)
#        # f.write(bytes(s, "utf-8"))
#        f.write(six.b(s))
#    else:
#        json.dump(d, f, indent=2)
#    return
#
