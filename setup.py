import sys
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

import numpy

if sys.platform == "linux":
    dirs_include = ["/opt/halcon/include/halconcpp/", "/opt/halcon/include/", "."]
    dirs_library = ["/opt/halcon/lib/x64-linux"]
elif sys.platform == "win32":
    dirs_include = ["C:/Program Files/MVTec/HALCON-13.0/include/halconcpp", "C:/Program Files/MVTec/HALCON-13.0/include", numpy.get_include(), "."]
    dirs_library = ["C:/Program Files/MVTec/HALCON-13.0/lib/x64-win64"]


setup(
    name="python-hcn",
    description="python bindings to Halcon proprietary vision system",
    author="Olivier Roulet-Dubonnet",
    version="0.1",
    license="LGPL",
    packages=["hcn"],
    ext_modules=cythonize([
        Extension(
            name="hcn.hcn",
            sources=["hcn\hcn.pyx", "hcn\cy_handler.cpp"],
            include_dirs=dirs_include,
            libraries=["halconcpp"],
            library_dirs=dirs_library,
            language="c++")
    ]))
