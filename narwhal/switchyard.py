""" Parser for the unfortunately difficult to parse test format used by the
Arctic Switchyard project. Maybe I'm being dumb and there's a straightforward
way to do this, but as far as I can tell, the format was constructed as a
competitive programming exercise. """

import dateutil.parser

def _read_headline(s):
    """ Parse a string like

    Cast 3  Station UW03   85.128 degrees North  _  045.068 degrees West   2013-5-2 / 1445 UTC

    """
    def calcdegrees(n, d, m, s):
        # Choose whether value n goes into degrees, minutes, or seconds
        if d is None: d = n
        elif m is None: m = n
        elif s is None: s = n
        else: raise ParseError()
        return d, m, s

    # Cast
    i = s.find("Cast") + 4
    cc = []
    cast = -1
    while cast == -1:
        c = s[i]
        if c.isdigit():
            cc.append(c)
        elif len(cc) != 0:
            cast = int("".join(cc))
        i += 1
        if i == 80:
            raise ParseError()

    # Station
    i = s.find("Station") + 7
    cc = []
    stn = -1
    while stn == -1:
        c = s[i]
        if c.isdigit():
            cc.append(c)
        elif len(cc) != 0:
            stn = int("".join(cc))
        i += 1
        if i == 80:
            raise ParseError()

    # Latitude
    cc = []
    hemisphere = None
    latdeg, latmin, latsec = None, None, None
    isnumeric = False
    cc = []
    while hemisphere is None:
        c = s[i]
        if c.isdigit() or c in "-.":
            isnumeric = True
            cc.append(c)
        elif c.isalpha():
            isnumeric = False
            cc.append(c)
        else:
            if isnumeric:
                (latdeg, latmin, latsec) = calcdegrees(float("".join(cc)), latdeg, latmin, latsec)
            else:
                s_ = "".join(cc).upper()
                if s_.startswith("N"):
                    hemisphere = 1
                if s_.startswith("S"):
                    hemisphere = -1
            cc = []

        i += 1
        if i == 80:
            raise ParseError()

    if latmin:
        latdeg += latmin / 60.0
    if latsec:
        latdeg += latsec / 3600.0
    latdeg *= hemisphere

    # Longitude
    cc = []
    hemisphere = None
    londeg, lonmin, lonsec = None, None, None
    isnumeric = False
    cc = []
    while hemisphere is None:
        c = s[i]
        if c.isdigit() or c in "-.":
            isnumeric = True
            cc.append(c)
        elif c.isalpha():
            isnumeric = False
            cc.append(c)
        else:
            if isnumeric:
                (londeg, lonmin, lonsec) = calcdegrees(float("".join(cc)), londeg, lonmin, lonsec)
            else:
                s_ = "".join(cc).upper()
                if s_.startswith("E"):
                    hemisphere = 1
                if s_.startswith("W"):
                    hemisphere = -1
            cc = []

        i += 1
        if i == 80:
            raise ParseError()

    if lonmin:
        londeg += lonmin / 60.0
    if lonsec:
        londeg += lonsec / 3600.0
    londeg *= hemisphere

    # Date
    while c != " ":
        i += 1
        c = s[i]
    dt = dateutil.parser.parse(s[i:])
    return cast, stn, latdeg, londeg, dt

s = "Cast 3  Station UW03        85.128 degrees North  _  045.068 degrees West         2013-5-2 / 1445 UTC"
print(_read_headline(s))

s = "Cast  1   Station 15        82 29.80 N 103 25.61 W      May 15 2011  17:21:10 UTC"
print(_read_headline(s))

s = """                    In Situ   Potential                     Sigma
   Depth    Pres     Temp      Temp      Cond    Salinity   -theta    Oxygen   Oxy Sat   Oxygen
    (m)   (dbar)    (degC)    (degC)    (S/m)     (psu)    (kg/m^3)   (ml/l)    (ml/l)  (umol/Kg)
"""

s2 = """                        In situ   Potential
     Depth     Pres     Temp      Temp      Cond    Salinity   Sigma     |----------Dissolved Oxygen---------|
      (m)     (dbar)    (degC)    (degC)    (S/m)     (psu)     -theta    (ml/l)   (mg/l)    (%sat)  (Mmol/Kg)
"""

# s = "   Depth    Pres     Temp      Temp      Cond    Salinity   -theta    Oxygen   Oxy Sat   Oxygen"

def _read_fields(s):
    """ Read a line of data fields.

    LDEO example:

                    In Situ   Potential                     Sigma
   Depth    Pres     Temp      Temp      Cond    Salinity   -theta    Oxygen   Oxy Sat   Oxygen
    (m)   (dbar)    (degC)    (degC)    (S/m)     (psu)    (kg/m^3)   (ml/l)    (ml/l)  (umol/Kg)

    UW example:

                        In situ   Potential
     Depth     Pres     Temp      Temp      Cond    Salinity   Sigma     |----------Dissolved Oxygen---------|
      (m)     (dbar)    (degC)    (degC)    (S/m)     (psu)     -theta    (ml/l)   (mg/l)    (%sat)  (Mmol/Kg)
    """
    spaces = 99
    columns = []
    cc = []
    ic = 0
    start = 0
    for c in s:

        if not c.isspace():

            if spaces > 0:
                spaces = 0

                if c == "(" and len(cc) != 0:
                    columns.append((start, ic, "".join(cc)))
                    cc = []

                if len(cc) == 0:
                    start = ic

            cc.append(c)

        else:
            spaces += 1
            if spaces == 2:
                columns.append((start, ic, "".join(cc)))
                cc = []

        ic += 1

    if len(cc) != 0:
        columns.append((start, len(s), "".join(cc)))
    return columns

field_cols = _read_fields(s)
print(field_cols)

def _construct_fields(cols):
    """ Combine fields read from multiple lines into a single field name by
    prepending the text from previous lines to the closest matching field. """

    def prepend_overlapping(col0, col1):
        names = []
        for (i0, i1, s) in col0:
            appended = False
            for (j0, j1, t) in col1:
                # Fields in consecutive rows are sort of aligned, plus or minus a bit
                if i0 <= j0 <= i1 or i0 <= j1 <= i1 or j0 <= i0 <= j1 or j0 <= i1 <= j1:
                    names.append((min(i0, j0), max(i1, j1), t+" "+s))
                    appended = True
                    break
            if not appended:
                names.append((i0, i1, s))
        return names

    names = cols[-1]
    for col in cols[-2::-1]:
        names = prepend_overlapping(names, col)
    return names

#_construct_fields(field_cols)

def parse_switchyard(fnm):
    """ Read a Switchyard CTD cast text file and return a dictionary containing
    the data and metadata. """

    with open(fnm) as f:
        line = f.readline()
        cnt = 0
        inheader = True
        d = {"comments": [],
             "longitude": None,
             "latitude": None,
             "time": None,
             "cast": None,
             "station": None,
             "fields": None,
             "data": []}
        field_cols = []

        while line != "":
            upline = line.strip().upper()
            if upline.startswith("CAST"):
                cast, stn, lat, lon, dt = _read_headline(line)
                d["cast"] = cast
                d["station"] = stn
                d["latitude"] = lat
                d["longitude"] = lon
                d["time"] = dt

            elif d["cast"] is None:
                d["comments"].append(line)

            elif inheader:
                if len(field_cols) != 0 and len(upline) <= 1:
                    inheader = False
                    d["fields"] = _construct_fields(field_cols)
                    nfields = len(d["fields"])
                else:
                    cols = _read_fields(line)
                    field_cols.append(cols)

            else:
                row = [float(a) for a in line.split()]
                if len(row) != nfields:
                    raise ParseError("cannot match fields to entries\n{0}\n{1}".format([a[2] for a in d["fields"]], row))
                else:
                    d["data"].append(row)

            cnt += 1
            line = f.readline()
    return d

class ParseError(Exception):
    pass
