from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

extensions=[Extension("thirdorder_core",
                      ["thirdorder_core.pyx"],
                      libraries=["symspg"],
                      include_dirs=[numpy.get_include()])]

setup(
    name="thirdorder",
    ext_modules=cythonize(extensions)
    )
