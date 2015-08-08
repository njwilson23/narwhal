#import datetime
import h5py

def write(fnm, d):
    f = h5py.File(fnm, "w")
    dicttogroup(d, f)
    return

def dicttogroup(d, h5group):
    for k, v in d.items():
        if isinstance(v, dict):
            g = h5group.create_group(k)
            dicttogroup(v, g)
        else:
            h5group[k] = v
    return

def read(fnm):
    f = h5py.File(fnm, "r")
    return grouptodict(f, dict())

def grouptodict(h5group, d):
    """ Recursively add branches from a h5py Group to a dictionary """
    for name in h5group:
        if isinstance(h5group[name], h5py.Group):
            d[name] = grouptodict(h5group[name], dict())
        else:
            d[name] = h5group[name]
    return d

