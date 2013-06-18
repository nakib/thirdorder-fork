# thirdorder #

The purpose of thirdorder.py is to help users of [wlsBTE](https://bitbucket.org/sousaw/wlsbte/) create FORCE\_CONSTANTS\_3RD files in an efficient and convenient manner. More specifically, it performs two tasks. First, it is able to obtain an irreducible  set of atomic displacements from which to compute the full anharmonic interatomic force constant (IFC) matrix. thirdorder.py then saves the displaced supercells to POSCAR files that can be fed to VASP. Second, from the vasprun.xml files created by VASP from these inputs, this script reconstructs the full IFC matrix and writes it in the right format to FORCE\_CONSTANTS\_3RD.

# Compilation #

thirdorder.py is a Python script. It was developed using Python 2.7.3, but should work with slightly older versions. In addition to the modules in Python's standard library, the numpy and scipy numerical libraries are required. Moreover, this script relies on a module, thirdorder\_core, which is written in a mixture of Cython and Fortran. Thus, in spite of Python being an interpreted language, a compilation step is needed.

An arch.make file is required for compilation. An example is provided with the distribution under the name arch.make.example, with the following contents:

```bash
export FC=gfortran
export FFLAGS=-g -O2 -fPIC  -fbounds-check 
export CFLAGS=-I/home/user/local/include
export LDFLAGS=-L/home/user/local/lib -llapack -lgfortran
```

The first line specifies the name of the Fortran compiler, in this case GNU Fortran, and the second lists the flags to be passed to it for compiling Fortran code. Especially relevant here is -fPIC, which ensures that the resulting object file can be linked into a dynamic library. The remaining two lines are flags to be passed to the C compiler in the compilation and linking stages, respectively. thirdorder.py uses Atsushi Togo's [spglib](http://spglib.sourceforge.net/), which must be available both at compilation and run time: make sure to include the pertinent -L flag among LDFLAGS, and to specify the path to libsymspg.so in your LD\_LIBRARY\_PATH environment variable. Finally, note that -lgfortran is needed when using gfortran.

Once arch.make is ready, thirdorder\_core.py can be compiled simply by running make from the root directory of the distribution.

# Usage #

After a successful compilation, the python subdirectory will contain two files, thirdorder\_core.so and thirdorder.py. Both are needed to run the script. You can either use them from that directory (maybe including it in your PATH for convenience) or copy thirdorder.py to a directory in your PATH and thirdorder\_core.so to any location where Python can find it for importing.

Any invocation of thirdorder.py requires a POSCAR file with a description of the unit cell to be present in the current directory. The script uses no other configuration files, and takes exactly five mandatory command-line arguments:

```
thirdorder.py sow|reap na nb nc cutoff[nm/-integer]
```

The first argument must be either "sow" or "reap", and chooses the operation to be performed (displacement generation or IFC matrix reconstruction). The next three must be positive integers, and specify the dimensions of the supercell to be created. Finally, the "cutoff" parameter decides on a force cutoff distance. Interactions between atoms spaced further than this parameter are neglected. If cutoff is a positive real number, it is interpreted as a distance in nm; on the other hand, if it is a negative integer -n, the maximum distance among n-th neighbors in the supercell is automatically determined and the cutoff distance is set accordingly.

# An example #

The following POSCAR describes the relaxed geometry of the primitive unit cell of InAs, a III-V semiconductor with a zincblende structure:

```
InAs
   6.00000000000000
     0.0000000000000000    0.5026468896190005    0.5026468896190005
     0.5026468896190005    0.0000000000000000    0.5026468896190005
     0.5026468896190005    0.5026468896190005    0.0000000000000000
   In   As
   1   1
Direct
  0.0000000000000000  0.0000000000000000  0.0000000000000000
  0.2500000000000000  0.2500000000000000  0.2500000000000000
```

Let us asuume that such POSCAR is in the current directory and that thirdorder.py is in our PATH. To generate an irreducible set of displacements for a 4x4x4 supercell and up-to-third-neighbor interactions, we run

```
thirdorder.py sow 4 4 4 -3
```

This creates a file called 3RD.SPOSCAR with the undisplaced supercell coordinates and 144 files with names following the pattern 3RD.POSCAR.*. It is the latter that need to be input to VASP. This step is completely system-dependent, but suppose that in ~/vaspinputs we have the required INCAR, POTCAR and KPOINTS files as well as a runvasp.sh script that can be passed to qsub. We can run a command sequence like:

```bash
for i in 3RD.POSCAR.*;do
   s=$(echo $i|cut -d"." -f3)
   d=job-$s
   mkdir $d
   cp $i $d/POSCAR
   cp ~/vaspinputs/INCAR ~/vaspinputs/POTCAR ~/vaspinputs/KPOINTS $d
   cp ~/vaspinputs/runvasp.sh $d
   pushd $d
   qsub runvasp.sh
   popd
done
```

Some time later, after all these jobs have finished successfully, we only need to feed all the vasprun.xml files in the right order to thirdorder.py, this time in reap mode:

```
find job* -name vasprun.xml|sort -n|thirdorder.py reap 4 4 4 -3
```

If everything goes according to plan, a FORCE\_CONSTANTS\_3RD file will be created at the end of this run. Naturally, it is important to choose the same parameters for the sow and reap steps.
