import copy
import numpy as np
from narwhal.cast import Cast

def ccmeans(cc):
    """ Calculate a mean Cast along isopycnals from a CastCollection. """
    c0 = cc[0]
    s0 = c0["sigma"]
    sharedkeys = set(c0.data.keys()).intersection(
                    *[set(c.data.keys()) for c in cc[1:]]).difference(
                    set(("pres", "botdepth", "time")))

    data = dict()
    for key in sharedkeys:
        arr = np.empty((len(c0["pres"]), len(cc)))
        arr[:,0] = c0[key]
        for j, c in enumerate(cc[1:]):
            s = np.convolve(c["sigma"], np.ones(3)/3.0, mode="same")
            arr[:,j+1] = np.interp(s0, s, c[key])
        data[key] = np.nanmean(arr, axis=1)
    
    return Cast(copy.copy(c0["pres"]), **data)

def ccmeanp(cc):
    if False in (np.all(cc[0]["pres"] == c["pres"]) for c in cc):
        raise ValueError("casts must share pressure levels")
    p = cc[0]["pres"]
    data = dict()
    # shared keys are those in all casts, minus pressure and botdepth
    sharedkeys = set(cc[0].data.keys()).intersection(
                    *[set(c.data.keys()) for c in cc[1:]]).difference(
                    set(("pres", "botdepth", "time")))
    for key in sharedkeys:
        arr = np.vstack([c.data[key] for c in cc])
        data[key] = np.nanmean(arr, axis=0)

    return Cast(p, **data)

