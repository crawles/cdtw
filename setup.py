from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules = cythonize("cdtw.pyx"),
    include_dirs=[numpy.get_include()]
)

setup(
    ext_modules = cythonize("cdtw_window.pyx"),
    include_dirs=[numpy.get_include()]
)

