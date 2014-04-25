#  thirdorder, help compute anharmonic IFCs from minimal sets of displacements
#  Copyright (C) 2012-2013 Wu Li <wu.li.phys2011@gmail.com>
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

# This file contains Cython wrappers allowing the relevant functions
# in spglib and also around the Fortran subroutines in
# thirdorder_core.f90 that need to be used from Python.
# Finally, the the reconstruction of the anharmonic interatomic
# constant set from the minimal set of constants is also implemented
# in this file for efficiency.

from libc.stdlib cimport malloc,free
from libc.math cimport round,fabs,sqrt

import numpy
import scipy
import scipy.linalg

cimport cython
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
    Wrapper around wedge() that returns a python dictionary with all
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
    # The dictionary contains numpy arrays instead of the original
    # low-level ones. The following fragment handles these
    # assignations. One-based Fortran indices are converted to the
    # C/Python zero-based convention.
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
    nruter["ALLEquiList"]=nruter["ALLEquiList"].T-1
    nruter["TransformationArray"]=nruter["TransformationArray"].T
    nruter["IndependentBasis"]=nruter["IndependentBasis"].T-1
    return nruter


@cython.boundscheck(False)
def reconstruct_ifcs(phipart,wedgeres,list4,poscar,sposcar):
    """
    Recover the full anharmonic IFC set from the irreducible set of
    force constants and the information obtained from wedge().
    """
    cdef int ii,jj,ll,mm,nn,kk,ss,tt,ix
    cdef int nlist,nnonzero,natoms,ntot,tribasisindex,rowindex
    cdef numpy.ndarray nruter,naccumindependent,aa,bb,aux,ind1equi,ind2equi
    cdef numpy.ndarray Q,R,P,ones,multiplier,compensation,aphilist
    cdef int[:,:,:] vind1
    cdef int[:,:,:] vind2
    cdef double[:,:] vaa
    cdef double[:,:,:,:] doubletrans
    cdef double[:,:,:,:,:,:] vnruter

    nlist=wedgeres["Nlist"]
    natoms=len(poscar["types"])
    ntot=len(sposcar["types"])
    nruter=numpy.zeros((3,3,3,natoms,ntot,ntot))
    naccumindependent=numpy.insert(numpy.cumsum(wedgeres["NIndependentBasis"]),
                                   0,[0])
    ntotalindependent=naccumindependent[-1]
    for i,e in enumerate(list4):
        nruter[e[2],e[3],:,e[0],e[1],:]=phipart[:,i,:]
    philist=[]
    for ii in xrange(nlist):
        for jj in xrange(wedgeres["NIndependentBasis"][ii]):
            ll=wedgeres["IndependentBasis"][jj,ii]//9
            mm=(wedgeres["IndependentBasis"][jj,ii]%9)//3
            nn=wedgeres["IndependentBasis"][jj,ii]%3
            philist.append(nruter[ll,mm,nn,
                                  wedgeres["List"][0,ii],
                                  wedgeres["List"][1,ii],
                                  wedgeres["List"][2,ii]])
    aphilist=numpy.array(philist)
    ind1equi=-numpy.ones((natoms,ntot,ntot),dtype=numpy.int32)
    ind2equi=-numpy.ones((natoms,ntot,ntot),dtype=numpy.int32)
    vind1=ind1equi
    vind2=ind2equi
    for ii in xrange(nlist):
        for jj in xrange(wedgeres["Nequi"][ii]):
            vind1[wedgeres["ALLEquiList"][0,jj,ii],
                  wedgeres["ALLEquiList"][1,jj,ii],
                  wedgeres["ALLEquiList"][2,jj,ii]]=ii
            vind2[wedgeres["ALLEquiList"][0,jj,ii],
                  wedgeres["ALLEquiList"][1,jj,ii],
                  wedgeres["ALLEquiList"][2,jj,ii]]=jj
    aa=numpy.zeros((natoms*ntot*27,ntotalindependent))
    vaa=aa
    vtrans=wedgeres["TransformationArray"]
    nnonzero=0
    for ii in xrange(natoms):
        for jj in xrange(ntot):
            for ll in xrange(3):
                for mm in xrange(3):
                    for nn in xrange(3):
                        tribasisindex=(ll*3+mm)*3+nn
                        rowindex=(ii*natoms+jj)*27+tribasisindex
                        for kk in xrange(ntot):
                            for ix in xrange(nlist):
                                if vind1[ii,jj,kk]==ix:
                                    for ss in xrange(naccumindependent[ix],naccumindependent[ix+1]):
                                        tt=ss-naccumindependent[ix]
                                        vaa[rowindex,ss]+=vtrans[tribasisindex,tt,
                                                                 vind2[ii,jj,kk],ix]
                        vaa[nnonzero,:]=vaa[rowindex,:]
                        nnonzero+=1
    aux=aa[:nnonzero,:].T

    Q,R,P=scipy.linalg.qr(aux,mode="economic",pivoting=True)
    nnonzero=(numpy.abs(numpy.diag(R))>=1e-12).sum()

    bb=numpy.array(Q[:,:nnonzero])
    D=numpy.diag(aphilist)
    ones=numpy.ones_like(aphilist)
    bb=numpy.dot(D,bb)
    multiplier=-scipy.linalg.lstsq(bb,ones)[0]
    compensation=numpy.dot(D,numpy.dot(bb,multiplier))
    aphilist+=compensation

    # Build the final, full set of anharmonic IFCs.
    vnruter=nruter
    vnruter[...]=0.
    for ii in xrange(nlist):
        for jj in xrange(wedgeres["Nequi"][ii]):
            for ll in xrange(3):
                for mm in xrange(3):
                    for nn in xrange(3):
                        tribasisindex=(ll*3+mm)*3+nn
                        for ix in xrange(wedgeres["NIndependentBasis"][ii]):
                            vnruter[ll,mm,nn,wedgeres["ALLEquiList"][0,jj,ii],
                                    wedgeres["ALLEquiList"][1,jj,ii],
                                    wedgeres["ALLEquiList"][2,jj,ii]
                                    ]+=wedgeres["TransformationArray"][
                                        tribasisindex,ix,jj,ii]*aphilist[
                                            naccumindependent[ii]+ix]
    return nruter
