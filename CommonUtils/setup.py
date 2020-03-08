from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

#Setup build script, include numpy headers
setup(ext_modules=cythonize("RUtils.pyx"), include_dirs=[numpy.get_include()] )
