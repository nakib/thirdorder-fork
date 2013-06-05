#  thirdorder, help compute anharmonic IFCs from minimal sets of displacements
#  Copyright (C) 2012-2013 Wu Li <wu.li@cea.fr>
#  Copyright (C) 2012-2013 Jesús Carrete Montaña <jcarrete@gmail.com>
#  Copyright (C) 2012-2013 Natalio Mingo Bisquert <natalio.mingo@cea.fr>
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.

from libc.stdlib cimport malloc,free
from libc.math cimport round,fabs,sqrt

import numpy
import scipy
import scipy.linalg

cimport numpy
numpy.import_array()
cimport cthirdorder_core

# NOTE: all indices used in this module are zero-based.

# Thin, specialized wrapper around spglib.
cdef class SymmetryOperations:
  """
  Object that contains all the interesting information about the
  crystal symmetry group of a set of atoms.
  """
  cdef public double[:,:] __lattvectors
  cdef public int[:] __types
  cdef public double[:,:] __positions
  cdef readonly str symbol
  cdef readonly double[:] __shift
  cdef readonly double[:,:] __transform
  cdef readonly double[:,:,:] __rotations
  cdef readonly double[:,:] __translations
  cdef readonly double[:] __norms
  cdef double c_latvectors[3][3]
  cdef int *c_types
  cdef double (*c_positions)[3]
  cdef int natoms,nsyms
  cdef double symprec

  property lattice_vectors:
      def __get__(self):
          return numpy.asarray(self.__lattvectors)
  property types:
      def __get__(self):
          return numpy.asarray(self.__lattvectors)
  property positions:
      def __get__(self):
          return numpy.asarray(self.__positions)
  property origin_shift:
      def __get__(self):
          return numpy.asarray(self.__shift)
  property transformation_matrix:
      def __get__(self):
          return numpy.asarray(self.__transform)
  property rotations:
      def __get__(self):
          return numpy.asarray(self.__rotations)
  property translations:
      def __get__(self):
          return numpy.asarray(self.__translations)

  cdef void __build_c_arrays(self):
      """
      Build the internal low-level representations of the input
      parameters, ready to be passed to C functions.
      """
      self.c_types=<int*>malloc(self.natoms*sizeof(int))
      self.c_positions=<double(*)[3]>malloc(self.natoms*sizeof(double[3]))
      if self.c_types is NULL or self.c_positions is NULL:
          raise MemoryError()

  cdef void __refresh_c_arrays(self):
      """
      Copy the values of __types, __positions and __lattvectors to
      their C counterparts.
      """
      cdef int i,j
      for i in range(3):
          for j in range(3):
              self.c_latvectors[i][j]=self.__lattvectors[i,j]
      for i in range(self.natoms):
          self.c_types[i]=self.__types[i]
          for j in range(3):
              self.c_positions[i][j]=self.__positions[i,j]

  cdef void __spg_get_dataset(self) except *:
      """
      Thin, slightly selective wrapper around spg_get_dataset(). The
      interesting information is copied out to Python objects and the
      rest discarded.
      """
      cdef int i,j,k
      cdef double tmp
      cdef cthirdorder_core.SpglibDataset *data
      data=cthirdorder_core.spg_get_dataset(self.c_latvectors,
                                            self.c_positions,
                                            self.c_types,
                                            self.natoms,
                                            self.symprec)
      # The C arrays can get corrupted by this function call.
      self.__refresh_c_arrays()
      if data is NULL:
          raise MemoryError()
      self.symbol=data.international_symbol.encode("ASCII").strip()
      self.__shift=numpy.empty((3,),dtype=numpy.float64)
      self.__transform=numpy.empty((3,3),dtype=numpy.float64)
      self.nsyms=data.n_operations
      self.__rotations=numpy.empty((self.nsyms,3,3),
                                   dtype=numpy.float64)
      self.__translations=numpy.empty((self.nsyms,3),
                                      dtype=numpy.float64)
      for i in range(3):
          self.__shift[i]=data.origin_shift[i]
          for j in range(3):
              self.__transform[i,j]=data.transformation_matrix[i][j]
      for i in range(self.nsyms):
          for j in range(3):
              self.__translations[i,j]=data.translations[i][j]
              for k in range(3):
                  self.__rotations[i,j,k]=data.rotations[i][j][k]
      cthirdorder_core.spg_free_dataset(data)

  def __cinit__(self,lattvectors,types,positions,symprec=1e-5):
      cdef int i
      self.__lattvectors=numpy.array(lattvectors,dtype=numpy.float64)
      self.__types=numpy.array(types,dtype=numpy.int32)
      self.__positions=numpy.array(positions,dtype=numpy.float64)
      self.__norms=numpy.empty((3,),dtype=numpy.float64)
      for i in range(3):
          self.__norms[i]=sqrt(lattvectors[i,0]**2+
                               lattvectors[i,1]**2+
                               lattvectors[i,2]**2)
      self.natoms=self.positions.shape[0]
      self.symprec=symprec
      if self.__positions.shape[0]!=self.natoms or self.__positions.shape[1]!=3:
          raise ValueError("positions must be a natoms x 3 array")
      if not (self.__lattvectors.shape[0]==self.__lattvectors.shape[1]==3):
          raise ValueError("lattice vectors must form a 3 x 3 matrix")
      self.__build_c_arrays()
      self.__refresh_c_arrays()
      self.__spg_get_dataset()

  def __dealloc__(self):
      if self.c_types is not NULL:
          free(self.c_types)
      if self.c_positions is not NULL:
          free(self.c_positions)


def pywedge(poscar,sposcar,symops,frange):
    """
    Wrapper around wedge() thar returns a python dictionary with all
    relevant information about the irreducible displacements.
    """
    cdef double ForceRange
    cdef int Ngrid1,Ngrid2,Ngrid3,Nsymm,Natoms,Ntot,Nlist,Allocsize
    cdef double LatVec[3][3],(*Coord)[3],(*CoordAll)[3]
    cdef double (*Orth)[3][3],(*Trans)[3]
    cdef void *vNequi,*vList,*vALLEquiList,*vTransformationArray
    cdef void *vNIndependentBasis,*vIndependentBasis
    cdef int i,j,k

    crotations=numpy.empty_like(symops.rotations)

    ForceRange=frange
    Ngrid1=sposcar["na"]
    Ngrid2=sposcar["nb"]
    Ngrid3=sposcar["nc"]
    Nsymm=symops.translations.shape[0]
    Natoms=len(poscar["types"])
    Ntot=len(sposcar["types"])
    Coord=<double(*)[3]>malloc(Natoms*sizeof(double[3]))
    CoordAll=<double(*)[3]>malloc(Ntot*sizeof(double[3]))
    Orth=<double(*)[3][3]>malloc(Nsymm*sizeof(double[3][3]))
    Trans=<double(*)[3]>malloc(Nsymm*sizeof(double[3][3]))

    for i in range(Nsymm):
        crotations[i,:,:]=numpy.dot(
            scipy.linalg.solve(poscar["lattvec"].T,symops.rotations[i,:,:].T),
            poscar["lattvec"].T).T
    cpos=numpy.dot(poscar["lattvec"],poscar["positions"])
    cposall=numpy.dot(sposcar["lattvec"],sposcar["positions"])
    for i in range(3):
        for j in range(3):
            LatVec[j][i]=poscar["lattvec"][i,j]
    for i in range(Natoms):
        for j in range(3):
            Coord[i][j]=cpos[j,i]
    for i in range(Ntot):
        for j in range(3):
            CoordAll[i][j]=cposall[j,i]
    for i in range(Nsymm):
        for j in range(3):
            Trans[i][j]=symops.translations[i,j]
            for k in range(3):
                Orth[i][k][j]=crotations[i,j,k]
    cthirdorder_core.wedge(LatVec,Coord,CoordAll,Orth,Trans,Natoms,
                           &Nlist,&vNequi,&vList,
                           &vALLEquiList,&vTransformationArray,
                           &vNIndependentBasis,&vIndependentBasis,
                           Ngrid1,Ngrid2,Ngrid3,Nsymm,ForceRange,
                           &Allocsize)
    free(Trans)
    free(Orth)
    free(CoordAll)
    free(Coord)
    nruter=dict()
    nruter["Nlist"]=int(Nlist)
    nruter["Nequi"]=numpy.empty(Nlist,dtype=numpy.int32)
    nruter["Nequi"][:]=<int[:Nlist]>vNequi
    nruter["List"]=numpy.empty((Nlist,3),dtype=numpy.int32)
    nruter["List"][:,:]=<int[:Nlist,:3]>vList
    nruter["ALLEquiList"]=numpy.empty((Nlist,Nsymm*6,3),dtype=numpy.int32)
    nruter["ALLEquiList"][:,:,:]=<int[:Nlist,:Nsymm*6,:3]>vALLEquiList
    nruter["TransformationArray"]=numpy.empty((Nlist,Nsymm*6,27,27))
    nruter["TransformationArray"][:,:,:,:]=(<double[:Nlist,:Nsymm*6,:27,:27]>
                                            vTransformationArray)
    nruter["NIndependentBasis"]=numpy.empty(Nlist,dtype=numpy.int32)
    nruter["NIndependentBasis"][:]=<int[:Nlist]>vNIndependentBasis
    nruter["IndependentBasis"]=numpy.empty((Nlist,27),dtype=numpy.int32)
    nruter["IndependentBasis"][:,:]=<int[:Nlist,:27]>vIndependentBasis
    cthirdorder_core.free_wedge(Allocsize,Nsymm,vNequi,vList,vALLEquiList,
                                vTransformationArray,vNIndependentBasis,
                                vIndependentBasis)
    nruter["List"]=nruter["List"].T-1
    nruter["AllEquiList"]=nruter["ALLEquiList"].T-1
    nruter["TransformationArray"]=nruter["TransformationArray"].T
    nruter["IndependentBasis"]=nruter["IndependentBasis"].T-1
    return nruter
