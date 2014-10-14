import copy
import itertools
from functools import reduce
import numpy as np
from narwhal.cast import AbstractCast, AbstractCastCollection

def count_casts(castlikes):
    """ Given an iterable of Casts and CastCollections, count the number of
    individual Casts. """
    n = 0
    for c in castlikes:
        if isinstance(c, AbstractCast):
            n += 1
        else:
            n += count_casts(c)
    return n

def ensureiterable(item):
    """ Turn *item* into an infinite lazy iterable. """
    if not hasattr(item, "__iter__") or isinstance(item, str):
        return itertools.repeat(item)
    else:
        return itertools.cycle(item)

def getiterable(kw, name, default):
    """ Equivalent to dict.get except that it ensures the result is an iterable. """
    return ensureiterable(kw.get(name, default))

def _nanmean(arr, axis=0):
    """ Re-implement nanmean in a way that doesn't fire off warning when there
    are NaN-filled rows. 
    
    Note that here axis is the axis to retain, which is not the behaviour of
    np.nanmean. I did it this way for simplicity, but this might be confusing
    in the future.
    """
    if axis != 0:
        arr = np.rollaxis(arr, axis)
    means = np.empty(arr.shape[0], dtype=arr.dtype)
    i = 0
    for row in arr:
        valid = row[~np.isnan(row)]
        if len(valid) > 0:
            means[i] = np.mean(row[~np.isnan(row)])
        else:
            means[i] = np.nan
        i += 1
    return means

def ccmeans(cc):
    """ Calculate a mean Cast along isopycnals from a CastCollection. """
    c0 = max(cc, key=lambda c: c.nvalid())
    s0 = c0["sigma"]
    sharedkeys = set(c0.data.keys()).intersection(
                    *[set(c.data.keys()) for c in cc[1:]]).difference(
                    set(("pres", "botdepth", "time")))
    nanmask = reduce(lambda a,b: a*b, [c.nanmask() for c in cc])
    data = dict()
    for key in sharedkeys:
        arr = np.nan * np.empty((len(cc), len(c0["pres"])))
        arr[0,:] = c0[key]
        for j, c in enumerate(cc[1:]):
            s = np.convolve(c["sigma"], np.ones(3)/3.0, mode="same")
            arr[j+1,:] = np.interp(s0, s, c[key])
        data[key] = _nanmean(arr, axis=1)
        data[key][nanmask] = np.nan

    return Cast(copy.copy(c0["pres"]), **data)

def ccmeanp(cc):
    if False in (np.all(cc[0]["pres"] == c["pres"]) for c in cc):
        raise ValueError("casts must share pressure levels")
    p = cc[0]["pres"]
    # shared keys are those in all casts, minus pressure and botdepth
    sharedkeys = set(cc[0].data.keys()).intersection(
                    *[set(c.data.keys()) for c in cc[1:]]).difference(
                    set(("pres", "botdepth", "time")))
    data = dict()
    for key in sharedkeys:
        arr = np.vstack([c.data[key] for c in cc])
        data[key] = _nanmean(arr, axis=1)

    return Cast(p, **data)

