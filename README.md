# Narwhal

Narwhal is a library for organizing and manipulating hydrographic data.

|   Data type           |   Narwhal class             |
|-----------------------|-----------------------------|
|   CTD cast            |   `cast.Cast`               |
|   Multiple casts      |   `cast.CastCollection`     |
|   Bathymetric line    |   `bathymetry.Bathymetry2d` |

Usefully, the `Cast` and `CastCollection` types can also serialize themselves as
JSON objects for storage.

Narwhal wraps the Gibbs Seawater Toolbox
([](http://www.teos-10.org/pubs/gsw/html/gsw_contents.html)) in the
`narwhal.gsw` submodule.

## Dependencies
- numpy
- scipy
- C-compiler
- matplotlib
- requests

Narwhal requires [karta](https://github.com/njwilson23/karta). The
`narwhal.plotting` module contains functions for displaying narwhal data with
matplotlib.

