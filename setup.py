from __future__ import print_function
import os, shutil
from distutils.core import setup
from distutils.extension import Extension
#from numpy.distutils.extension import Extension
import install_gsw

try:
    print("Fetching GSW v3.02", end="")
    install_gsw.download_zip("http://www.teos-10.org/software/gsw_c_v3.02.zip",
                             fnm="temp_gsw_c.zip")

    if install_gsw.compare_md5("temp_gsw_c.zip", 
                               "6360ec9cff432f7bc01032fbecf48422"):
        install_gsw.unzip("temp_gsw_c.zip", "deps/")
        ext = [Extension("narwhal.cgsw",
                         sources=["deps/gsw_c_v3.02/gsw_oceanographic_toolbox.c",
                                  "deps/gsw_c_v3.02/gsw_saar.c"], 
                         include_dirs=["deps/gsw_c_v3.02/"])]

    else:
        raise Exception("MD5 for downloaded GSW source doesn't match "
                        "expected digest. GSW will not be installed.")
    os.remove("temp_gsw_c.zip")
    print("...done")

except Exception as e:
    print("Failed to download and install Gibbs Seawater Toolbox")
    print(e)
    ext = []

setup(
    name = "narwhal",
    version = "0.1b",
    author = "Nat Wilson",
    #package_dir = {"narwhal": "src"},
    packages = ["narwhal"],
    ext_modules = ext,
)

