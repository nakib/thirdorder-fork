from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

setup(
    cmdclass = {'build_ext':build_ext},
    ext_modules = [Extension("thirdorder_core",
                             ["thirdorder_core.pyx","cthirdorder_core.pxd"],
                             libraries=["symspg"],
                             extra_objects=["thirdorder_fortran.o"])]
    )
