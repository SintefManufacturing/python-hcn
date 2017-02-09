import sys
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

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
            sources=["hcn/hcn.pyx"],
            include_dirs=["/opt/halcon/include/halconcpp/", "/opt/halcon/include/"],
            libraries=["halconcpp"],
            library_dirs=["/opt/halcon/lib/x64-linux"],
            language="c++")
    ]))
