# Narwhal

[![Build Status](https://travis-ci.org/njwilson23/narwhal.svg?branch=master)](https://travis-ci.org/njwilson23/narwhal)

Narwhal is a Python2 / Python3 library for organizing and manipulating hydrographic data.

![](classmap.svg)

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

Narwhal is experimental, and should be considered alpha quality. Also consider
[python-oceans](https://github.com/ocefpaf/python-oceans) and
[oce](https://github.com/dankelley/oce) (R).

