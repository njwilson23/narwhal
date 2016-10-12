# Narwhal

[![Build Status](https://travis-ci.org/njwilson23/narwhal.svg?branch=master)](https://travis-ci.org/njwilson23/narwhal)

## Oceanographic data analysis in Python

Narwhal is a Python module built on [pandas](http://pandas.pydata.org/) and
[matplotlib](http://matplotlib.org/). Narwhal is designed for manipulating and
visualizing oceanographic data.

Data are organized into self-describing `Cast` and `CastCollection` data
structures. Convenience methods and functions are included for:

- interpolation
- density and depth calculation
- buoyancy frequency estimation
- baroclinic mode analysis
- water type fraction inversion

### Quickly visualize results

The `narwhal.plotting` submodule contains convenience methods for creating T-S
diagrams, cast plots, and section plots. Here's some data from off the coast of
northeastern Greenland:

![T-S diagram](https://github.com/njwilson23/narwhal/raw/gh-pages/images/ts-plot2_v7.png)

![Section diagram](https://github.com/njwilson23/narwhal/raw/gh-pages/images/sill_velocity.png)

### Python wrapper for the thermodynamic equation of state

Narwhal provides a *ctypes* wrapper for the
[Gibbs Seawater Toolbox](http://www.teos-10.org/pubs/gsw/html/gsw_contents.html)
in the `narwhal.gsw` submodule, making things like the following possible:

    density = narwhal.gsw.rho(cast["sa"], cast["ct"], cast["p"])

Currently, GSW 3.05 is packaged with Narwhal.

### Data should not be tied to software

For storage, data is serialized to JSON or HDF files. These common formats are
open and easily imported into other analysis packages (such as MATLAB), or
visualization libraries (such as D3).

## Installation

    git clone https://github.com/njwilson23/narwhal.git
    pip install -r narwhal/requirements.txt
    pip install narwhal

### Dependencies

- Python 2.7+ or Python 3.4+
- pandas
- matplotlib
- scipy
- requests
- dateutil
- six
- C-compiler (for GSW)
- h5py (optional, required for HDF read/write)

If [Karta](https://github.com/fortyninemaps/karta) is installed, it will be used
for fast and accurate geographical calculations.

Narwhal is experimental. See also
[python-oceans](https://github.com/ocefpaf/python-oceans) and
[oce](https://github.com/dankelley/oce) (R).
