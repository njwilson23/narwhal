""" Experimental functions for interpolating ocean properties in a vertical section.

The following assumptions are made:
    - Variability on veritical scales is much greater than on horizontal scales
    - Topographic features block mixing

Additional functions will take a Bathymetry instance as a constraint, or be
designed to interpolate velocity data, which is less stratified
"""

import threading
from scipy.interpolate import Rbf


def create_2d_interpolator(x, z, d, xscale, zscale, **kw):
    def norm(x1, x2):
        return np.sqrt(((x1-x2)**2 \
                * np.array([[[1/xscale**2, 1/zscale**2]]]).swapaxes(0, 2))\
                .sum(axis=0))
    return Rbf(x, z, d, norm=norm, **kw)

def partition_by_trough(xi, zi, xb, depths):
    """ Partition a grid defined by vector *xi* and *zi* based on bathymetry *depths* measured at *xb*. """
    # Partition interpolation domain
    interp_regions = []
    marked = np.zeros([len(zi), len(xi)], dtype=np.bool)   # points that have been covered

    for i, zs in sorted(enumerate(depths), key=lambda a: a[1]):
        xleft = -0.1,
        xright = xb[-1]

        # Search for neighbouring constraining sills
        for j in range(i, len(depths)):
            if depths[j] < zs:
                xright = xb[j]

        for j in range(i, -1, -1):
            if depths[j] < zs:
                xleft = xb[j]

        region = [j*len(xi)+i for j,z_ in enumerate(zi)
                              for i,x_ in enumerate(xi)
                              if (~marked[j,i]) and (z_ <= zs) and (xleft < x_ <= xright)]
        
        if len(region) != 0:
            interp_regions.append(region)
            for i in interp_regions[-1]:
                i_ = i%len(xi)
                j_ = i//len(xi)
                marked[j_,i_] = True
    return interp_regions

def interp_over_sills(casts, prop, axiskey, xscale=10e3, zscale=10,
                      nz=20, nx=4, **kw):

    xp = casts.projdist()

    # Create arrays of measured locations, data, and maximum depth
    x = []
    z = []
    d = []
    maxdepths = []
    for i, cast in enumerate(casts):
        nm = cast.nanmask([prop, axiskey])
        d.extend(cast[prop][~nm])
        x.extend(xp[i]*np.ones(sum(~nm)))
        z_ = cast[axiskey][~nm]
        z.extend(z_)
        maxdepths.append(z_.max())
        
    xo = np.array(x)
    zo = np.array(z)
    do = np.array(d)

    # Generate interpolation points (can replace this code later)
    xi = np.interp(np.arange(len(casts)*nx), np.arange(len(casts))*nx, xp)
    zi = np.linspace(0.0, max(maxdepths), nz)

    # Partition interpolation domain
    interp_regions = partition_by_trough(xi, zi, xp, maxdepths)
    print("Computing over %i regions" % len(interp_regions))

    # Start interpolation thread at each region
    threads = []
    results = []
    for region in interp_regions:
        xi_ = np.array([xi[idx%len(xi)] for idx in region])
        zi_ = np.array([zi[idx//len(xi)] for idx in region])

        # Find the relevant data points
#         xo_ = xo[(xo >= xi_.min()) & (xo <= xi_.max()) & (zo <= zi_.max())]
#         zo_ = zo[(xo >= xi_.min()) & (xo <= xi_.max()) & (zo <= zi_.max())]
#         do_ = do[(xo >= xi_.min()) & (xo <= xi_.max()) & (zo <= zi_.max())]
        xo_ = xo[(xo >= xi_.min()) & (xo <= xi_.max())]
        zo_ = zo[(xo >= xi_.min()) & (xo <= xi_.max())]
        do_ = do[(xo >= xi_.min()) & (xo <= xi_.max())]
        
        # Debugging
#         xo_ = xo
#         zo_ = zo
#         do_ = do

        fint = create_2d_interpolator(xo_, zo_, do_, xscale, zscale, **kw)
        
        def fthread(result):
            result[:] = fint(xi_, zi_)
            return

        result = np.zeros(len(xi_))
        results.append(result)

        th = threading.Thread(target=fthread, args=(result,))
        threads.append(th)
        th.start()

    for th in threads:
        th.join()

    # Assemble product
    combined_result = np.zeros(len(xi)*len(zi))
    for region, res in zip(interp_regions, results):
        for i,d in zip(region, res):
            combined_result[i] = d
    return xi, zi, combined_result.reshape([len(zi), len(xi)]), interp_regions

