#import datetime
import numpy as np
import h5py

def debug_showtypes(d, indent=0):
    """ Graph a narwhal serialization dictionary. """
    sp = " "*indent
    for k,v in d.items():
        if isinstance(v, dict):
            print(sp, k, "->")
            debug_showtypes(v, indent=indent+2)
        elif isinstance(v, list):
            print(sp, k, "-> [")
            for i, item in enumerate(v):
                if isinstance(item, dict):
                    debug_showtypes(item, indent+2)
                    print()
                elif i==0:
                    print(sp+"  ", type(item))
            print(sp, "]")
        elif isinstance(v, tuple):
            print(sp, k, "==", v)
        else:
            print(sp, k, "->", type(v))

def write(fnm, d):
    f = h5py.File(fnm, "w")
    f.attrs["hdf_driver_version"] = "0.4.0"
    dicttogroup(d, f)
    return

def dicttogroup(d, h5group):
    for k, v in d.items():
        if k == "casts":
            castgroup = h5group.create_group("casts")
            # HDF doesn't have a heterogeneous list concept
            # Create a level of Cast groups with integer keys
            for i, d in enumerate(v):
                g = castgroup.create_group(str(i))
                dicttogroup(d, g)
        elif isinstance(v, dict):
            g = h5group.create_group(k)
            dicttogroup(v, g)
        else:
            try:
                if isinstance(v, (list, np.ndarray)):
                    dset = h5group.create_dataset(k, data=v,
                                                     compression="gzip",
                                                     compression_opts=4)
                else:
                    h5group[k] = v
            except TypeError as e:
                print(k, str(e))
    return

def read(fnm):
    f = h5py.File(fnm, "r")
    return grouptodict(f, dict())

def grouptodict(h5group, d):
    """ Recursively add branches from a h5py Group to a dictionary """
    for name in h5group:
        if isinstance(h5group[name], h5py.Group):
            if name == "casts":     # Special case the list of Cast dicts
                d["casts"] = []
                for _, group in h5group[name].items():
                    d["casts"].append(grouptodict(group, dict()))
            else:
                d[name] = grouptodict(h5group[name], dict())
        else:
            d[name] = h5group[name].value
    return d
