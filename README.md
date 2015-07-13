# Narwhal

[![Build Status](https://travis-ci.org/njwilson23/narwhal.svg?branch=master)](https://travis-ci.org/njwilson23/narwhal)

Narwhal is a Python module built on [pandas](http://pandas.pydata.org/) and
[matplotlib](http://matplotlib.org/) for manipulating and visualizing
oceanographic survey data.

Oceanographic data are organized into `Cast` and `CastCollection` data structures. Functions are provided for

- interpolation
- density and depth calculation
- buoyancy frequency estimation
- baroclinic mode analysis
- water fraction inversion

More specialized techniques can be added by subclassing the generic
`Cast` type (see the `XBTCast` and `LADCP` classes as an example).

For storage, data is serialized as zipped JSON objects streams. Due to the
prevalence of JSON parsers, these data formats can be easily parsed into other
analysis packages, and are furthermore ready for web-based visualization.

Parsers are also included for WOCE CTD casts in NetCDF format and Arctic
Switchyard CTD casts in ASCII table format.

The `narwhal.plotting` submodule contains convenience methods for creating T-S
diagrams, cast plots, and section plots. Here's some data from the [WOCE P17N
line](http://cchdo.ucsd.edu/cruise/325021_1), collected on a cruise by the
Thomas G. Thomson.

![P17N T-S diagram](https://rawgit.com/njwilson23/narwhal/gh-pages/ts-demo.png)

![P17N section diagram](https://rawgit.com/njwilson23/narwhal/gh-pages/section-demo.png)

Narwhal provides a *ctypes* wrapper for the
[Gibbs Seawater Toolbox](http://www.teos-10.org/pubs/gsw/html/gsw_contents.html)
in the `narwhal.gsw` submodule, making things like the following possible:

    density = narwhal.gsw.rho(cast["abssal"], cast["constemp"], cast["pres"])

## Installation

    git clone https://github.com/njwilson23/narwhal.git
    pip install -r narwhal/requirements.txt
    pip install narwhal

## Dependencies

- Python 2.7+ or Python 3.2+
- pandas
- numpy
- matplotlib
- scipy
- [karta](https://github.com/njwilson23/karta)
- requests
- six
- C-compiler (for GSW)

Narwhal is experimental. See also
[python-oceans](https://github.com/ocefpaf/python-oceans) and
[oce](https://github.com/dankelley/oce) (R).

