from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

#Setup build script, include numpy headers
# python3 setup.py build_ext --inplace 
setup(ext_modules=cythonize("RUtils.pyx"), include_dirs=[numpy.get_include()] )
