
class Cast(object):
    """ A Cast is a set of pressure, salinity, temperature (et al)
    measurements associated with a single coordinate. """

    _type = "ctdcast"

    def __init__(self, p, S=None, T=None, coords=None, **kwargs):

        self.coords = coords

        def _fieldmaker(n, arg):
            if arg is not None:
                return arg
            else:
                return [None for _ in xrange(n)]

        self.data = {}
        self.data["pres"] = p
        self.data["sal"] = _fieldmaker(len(p), S)
        self.data["temp"] = _fieldmaker(len(p), T)

        for kw,val in kwargs.iteritems():
            self.data[kw] = _fieldmaker(len(p), val)

        self._len = len(p)
        self._fields = tuple(["pres", "sal", "temp"] + [a for a in kwargs])

        return

    def __getitem__(self, key):
        if isinstance(key, int):
            if key < self._len:
                return tuple(self.data[a][key] for a in self._fields)
            else:
                raise IndexError("Index ({0}) is greater than cast length ({1})".format(key, self._len))
        elif key in self.data:
            return self.data[key]
        else:
            raise KeyError("No item {0}".format(key))

class CastCollection(object):
    """ A CastCollection is an indexable collection of Cast instances """
    def __init__(self, *args):
        if isinstance(args[0], Cast):
            self.casts = args
        elif (len(args) == 1) and (False not in (isinstance(a, Cast) for a in args[0])):
            self.casts = args[0]
        else:
            raise TypeError("Arguments must be either Cast types or an "
                            "iterable collection of Cast types")
        return
