# Narwhal

[![Build Status](https://travis-ci.org/njwilson23/narwhal.svg?branch=master)](https://travis-ci.org/njwilson23/narwhal)

Narwhal is a Python2 / Python3 library for organizing and manipulating hydrographic data.

| Narwhal class             |                 | Description               |
|---------------------------|-----------------|---------------------------|
| `cast.Cast`               |                 | Generic cast type         |
|                           | `cast.CTDCast`  | CTD Cast                  |
|                           | `cast.XBTCast`  | XBT Cast                  |
|                           | `cast.LADCP`    | LADCP observation         |
| `cast.CastCollection`     |                 | Multiple casts            |
| `bathymetry.Bathymetry2d` |                 | Bathymetric line          |   

The `Cast` and `CastCollection` types serialize themselves as JSON objects for
storage.

Narwhal provides a wrapper for the
[Gibbs Seawater Toolbox](http://www.teos-10.org/pubs/gsw/html/gsw_contents.html)
in the `narwhal.gsw` submodule.

## Dependencies
- numpy
- scipy
- requests
- [karta](https://github.com/njwilson23/karta)
- matplotlib (optional; for plotting)
- C-compiler (optional; for GSW)

Narwhal is under development and should be considered alpha quality.

