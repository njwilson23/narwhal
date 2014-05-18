""" Script that downloads the Fortran Gibbs Seawater Toolbox from

    http://www.teos-10.org/software.htm#1

and compiles it with f2py.

Reference
---------

McDougall, T.J. and P.M. Barker, 2011: Getting started with TEOS-10 and the
Gibbs Seawater (GSW) Oceanographic Toolbox, 28pp., SCOR/IAPSO WG127, ISBN
978-0-646-55621-5
"""

from __future__ import print_function
import os
import requests
import zipfile
import hashlib
#import numpy.f2py

def download_zip(url, fnm):
    """ Attempt to download source from TEOS-10 and save it to a zip file. """
    r = requests.get(url)
    with open(fnm, "wb") as f:
        f.write(r.content)
    return r

def unzip(fnm, to):
    if zipfile.is_zipfile(fnm):
        zf = zipfile.ZipFile(fnm)
        if not os.path.isdir(to):
            os.makedirs(to)
        #zf.extract("gsw_check_functions.f90", path=to)
        #zf.extract("gsw_oceanographic_toolbox.f90", path=to)
        #zf.extract("gsw_data_v3_0.dat", path=to)
        zf.extract("gsw_c_v3.02/gsw_check_functions.c", path=to)
        zf.extract("gsw_c_v3.02/gsw_format.c", path=to)
        zf.extract("gsw_c_v3.02/gsw_oceanographic_toolbox.c", path=to)
        zf.extract("gsw_c_v3.02/gsw_saar.c", path=to)
        zf.extract("gsw_c_v3.02/gsw_saar_data.c", path=to)
        zf.extract("gsw_c_v3.02/gswteos-10.h", path=to)
        zf.extract("gsw_c_v3.02/gsw_data_v3_0.dat.gz", path=to)
    else:
        raise IOError("There was a problem unzipping {0}".format(fnm))

    return

def build_ext(fnm):
    with open(fnm, "r") as f:
        source = f.read()
    res = numpy.f2py.compile(source, modulename="gsw", source_fn="tmp.f90",
                             )#extra_args="-h gsw.pyf")
    os.remove("tmp.f90")
    return

def compare_md5(fnm, against):
    md5 = hashlib.md5()
    with open(fnm, "rb") as f:
        md5.update(f.read())
        md5hash = md5.hexdigest()
    if md5hash != against:
        return False
    else:
        return True

if __name__ == "__main__":
    #download_zip("http://www.teos-10.org/software/gsw_fortran_v3_02.zip",
    #             fnm="temp_gsw.zip")
    #unzip("temp_gsw.zip", "deps/")
    #os.remove("temp_gsw.zip")
    #build_ext("deps/gsw_oceanographic_toolbox.f90")

    download_zip("http://www.teos-10.org/software/gsw_c_v3.02.zip",
                 fnm="temp_gsw_c.zip")

    if not compare_md5("temp_gsw_c.zip", "6360ec9cff432f7bc01032fbecf48422"):
        raise Exception("MD5 doesn't match expected digest - aborting!")

    unzip("temp_gsw_c.zip", "deps/")
    os.remove("temp_gsw_c.zip")

