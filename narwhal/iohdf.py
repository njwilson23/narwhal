import datetime
import h5py
from karta import Point, geojson
from . import units

TIME_TYPES = (datetime.date, datetime.time)

def save_object(obj, fnm, **kw):
    """ Save a narwhal object to an HDF file. Valid kwargs are:

    verbose :   print warnings
    """
    f = h5py.File(fnm, "w")
    if obj._type == "cast":
        g = f.create_group("cast")
        cast_to_group(obj, g, **kw)
    elif obj._type == "castcollection":
        g = f.create_group("castcollection")
        for i,cast in enumerate(obj):
            gcast = g.create_group("cast%i" % i)
            cast_to_group(cast, gcast, **kw)
    else:
        raise TypeError("Unexportable type: {0}".format(type(obj)))
    f.close()
    return f

def cast_to_group(cast, group, verbose=True):
    """ Save a narwhal Cast object to an HDF group. """
    gdata = group.create_group("data")
    gdata["zunits"] = cast.zunits
    gdata["zname"] = cast.zname
    for col in cast.data.columns:
        gdata.create_dataset(col, data=cast.data[col].values)

    gprop = group.create_group("properties")
    for k, v in cast.properties.items():
        if isinstance(v, datetime.datetime):
            gprop[k] = v.isoformat(sep=" ")
        elif isinstance(v, TIME_TYPES):
            gprop[k] = v.isoformat()
        else:
            try:
                gprop[k] = v
            except TypeError:
                if verbose:
                    print("Unable to serialize property {0} = {1}".format(k, v))

    return (gdata, gprop)

def group_to_narwhal(group):
    return

def group_to_cast(group):
    return

def group_to_cast_collection(group):
    return


