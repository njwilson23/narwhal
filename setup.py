from __future__ import print_function
import os, shutil
from distutils.core import setup
from distutils.extension import Extension
import install_gsw

try:
    print("Fetching GSW v3.03", end="")
    install_gsw.download_zip("http://www.teos-10.org/software/gsw_c_v3.03.zip",
                             fnm="temp_gsw_c.zip")

    if install_gsw.compare_md5("temp_gsw_c.zip", "1317c63c36bb4ee4f438c573d5bea2db"):
        install_gsw.unzip("temp_gsw_c.zip", "deps/")
        ext = [Extension("narwhal.cgsw",
                         sources=["deps/gsw_c_v3.03/gsw_oceanographic_toolbox.c",
                                  "deps/gsw_c_v3.03/gsw_saar.c"], 
                         include_dirs=["deps/gsw_c_v3.03/"])]

    else:
        raise Exception("MD5 for downloaded GSW source doesn't match "
                        "expected digest. GSW will not be installed.")
    print("...done")

except Exception as e:
    print("\nFailed to download and install Gibbs Seawater Toolbox")
    print(e)
    ext = []

finally:
    if os.path.isfile("temp_gsw_c.zip"):
        os.remove("temp_gsw_c.zip")

setup(
    name = "narwhal",
    version = "0.3.0",
    author = "Nat Wilson",
    #package_dir = {"narwhal": "src"},
    packages = ["narwhal"],
    ext_modules = ext,
)

