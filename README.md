# Narwhal

[![Build Status](https://travis-ci.org/njwilson23/narwhal.svg?branch=master)](https://travis-ci.org/njwilson23/narwhal)

## Oceanographic data analysis in Python

Narwhal is a Python module built on [pandas](http://pandas.pydata.org/) and
[matplotlib](http://matplotlib.org/). Narwhal is designed for manipulating and
visualizing oceanographic data.

Data are organized into self-describing `Cast` and `CastCollection` data
structures. Analysis functions are included for

- interpolation
- density and depth calculation
- buoyancy frequency estimation
- baroclinic mode analysis
- water type fraction inversion

### Quickly visualize results

The `narwhal.plotting` submodule contains convenience methods for creating T-S
diagrams, cast plots, and section plots. Here's some data from the [WOCE P17N
line](http://cchdo.ucsd.edu/cruise/325021_1), collected on a cruise by the
Thomas G. Thomson.

![P17N T-S diagram](https://rawgit.com/njwilson23/narwhal/gh-pages/ts-demo.png)

![P17N section diagram](https://rawgit.com/njwilson23/narwhal/gh-pages/section-demo.png)

### Python wrapper for the thermodynamic equation of state

Narwhal provides a *ctypes* wrapper for the
[Gibbs Seawater Toolbox](http://www.teos-10.org/pubs/gsw/html/gsw_contents.html)
in the `narwhal.gsw` submodule, making things like the following possible:

    density = narwhal.gsw.rho(cast["sa"], cast["ct"], cast["p"])

Currently, GSW 3.05 is included.

### Data should not be tied to software

For storage, data is serialized to JSON or HDF files. These common formats are
open and easily imported into other analysis packages (such as MATLAB), or
browser-based visualization libraries (such as D3).

Parsers are also included for WOCE CTD casts in NetCDF format and Arctic
Switchyard CTD casts in ASCII table format.

## Installation

    git clone https://github.com/njwilson23/narwhal.git
    pip install -r narwhal/requirements.txt
    pip install narwhal

### Dependencies

- Python 2.7+ or Python 3.3+
- pandas
- matplotlib
- scipy
- h5py (optional, required for HDF read/write)
- requests
- dateutil
- six
- C-compiler (for GSW)

If [karta](https://github.com/njwilson23/karta) is installed, it will be used to
provide the geographical back-end.

Narwhal is experimental. See also
[python-oceans](https://github.com/ocefpaf/python-oceans) and
[oce](https://github.com/dankelley/oce) (R).
