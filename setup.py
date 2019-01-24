from distutils.core import setup
from distutils.extension import Extension

ext_gsw = Extension("narwhal.cgsw",
                    sources=["deps/gsw_c_v3.05/gsw_oceanographic_toolbox.c",
                             "deps/gsw_c_v3.05/gsw_saar.c"],
                    include_dirs=["deps/gsw_c_v3.05"])

setup(
    name = "narwhal",
    version = "0.4.0b3",
    author = "Nat Wilson",
    packages = ["narwhal", "narwhal.plotting"],
    ext_modules = [ext_gsw],
)
