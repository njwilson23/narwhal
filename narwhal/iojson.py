""" Module for handling the serialization of Cast- and CastCollection-like
objects to persistent files. """

import json
import gzip
import six
import karta

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

def write(f, d, binary=True):
    if binary:
        s = json.dumps(d, indent=0, cls=NumpyJSONEncoder)
        f.write(six.b(s))
    else:
        json.dump(d, f, indent=2, cls=NumpyJSONEncoder)
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
#def castcollection_as_geojson(cc):
#    castpoints = (karta.Point(c.coords, properties={"id":i})
#                  for i, c in enumerate(cc))
#    geojsonstring = karta.geojson.printFeatureCollection(castpoints)
#    return geojsonstring

