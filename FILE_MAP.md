# Files included in this package #

```
├── arch.make.example
├── core
│   ├── cthirdorder_core.pxd
│   ├── setup.py
│   ├── thirdorder_core.pyx
│   └── thirdorder_fortran.f90
├── FILE_MAP.md
├── LICENSE
├── Makefile
├── python
│   └── thirdorder.py
└── README.md
```

# Organization of the code #

The central part of the code is the thirdorder.py script, written in Python. thirdorder.py takes care of all high-level operations including input/output and command-line argument processing. It also handles supercell creation, automatic determination of cutoff radii and the translation of a list of irreducible anharmonic interatomic force constants into a set of DFT runs required to obtain them.

In contrast, the most computationally expensive parts of the calculation are implemented in Fortran and included in thirdorder\_fortran.f90. The two most important subroutines in that file are `wedge()` and `gaussian()`. `wedge()` can be considered the core of the algorithm implemented by this package. Its function is to harness the linear constraints imposed by point-group symmetry to obtain a minimal set of anharmonic interatomic force constants that need to be calculated. `gaussian()` is an auxiliary subroutine that performs the Gaussian elimination procedure required by `wedge()`.

In addition to the Fortran code contained in thirdorder\_fortran.f90, thirdorder.py also makes use of Atsushi Togo's [spglib](http://spglib.sourceforge.net/) to treat symmetries. spglib is written in C. In order to make the C and Fortran routines available to Python, a Cython wrapper is needed. The wrapper is included in files cthirdorder\_core.pxd and thirdorder\_core.pyx. The former contains a set of straightforward prototype declarations mimicking those in thirdorder\_fortran.f90 and in spglib's spglib.h header. In thirdorder\_core.pyx the relevant parts of spglib are wrapped in a SymmetryOperations class. Convenient Python wrappers for `wedge()` and `gaussian()` are also defined. In addition to these wrappers, the reconstruction of the anharmonic interatomic constant set from the minimal set of constants is also implemented in this file for efficiency. Notably, translational symmetry is enforced here.

# Other files #

* README.md: User documentation for thirdorder.py

* LICENSE: Copy of the GNU General Public License, version 3

* Makefile: Set of make rules for building the Python module from Cython and Fortran code. Most of the work is actually performed by setup.py

* arch.make.example: Machine-specific variables needed for the compilation. Must be copied to arch.make and customized appropriately

* setup.py: Python distutils script needed to build the low-level module by calling the compiler, the linker and so on

* FILE\_MAP.md: This file
