""" Script that downloads the Gibbs Seawater Toolbox in C from

    http://www.teos-10.org/software.htm#1

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
        zf.extract("gsw_c_v3.03/gsw_check_functions.c", path=to)
        zf.extract("gsw_c_v3.03/gsw_format.c", path=to)
        zf.extract("gsw_c_v3.03/gsw_oceanographic_toolbox.c", path=to)
        zf.extract("gsw_c_v3.03/gsw_saar.c", path=to)
        zf.extract("gsw_c_v3.03/gsw_saar_data.c", path=to)
        zf.extract("gsw_c_v3.03/gswteos-10.h", path=to)
    else:
        raise IOError("There was a problem unzipping {0}".format(fnm))

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
    download_zip("http://www.teos-10.org/software/gsw_c_v3.03.zip",
                 fnm="temp_gsw_c.zip")

    if not compare_md5("temp_gsw_c.zip", "1317c63c36bb4ee4f438c573d5bea2db"):
        raise Exception("MD5 doesn't match expected digest - aborting!")

    unzip("temp_gsw_c.zip", "deps/")
    os.remove("temp_gsw_c.zip")

