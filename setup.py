import sys
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

#sys.path.append("src")

setup(
        name="halpy",
        version="0.1",
        license="LGPL",

        ext_modules=cythonize([
            Extension(
                name="halpy",
                sources=["src/halpy.pyx"],
                include_dirs=["/opt/halcon/include/halconcpp/", "/opt/halcon/include/"],
                libraries=["halconcpp"],
                library_dirs=["/opt/halcon/lib/x64-linux"],
                language="c++"
            )
            ])
)
