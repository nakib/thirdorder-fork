#  thirdorder, help compute anharmonic IFCs from minimal sets of displacements
#  Copyright (C) 2012-2014 Wu Li <wu.li.phys2011@gmail.com>
#  Copyright (C) 2012-2014 Jesús Carrete Montaña <jcarrete@gmail.com>
#  Copyright (C) 2012-2014 Natalio Mingo Bisquert <natalio.mingo@cea.fr>
#  Copyright (C) 2014      Antti J. Karttunen <antti.j.karttunen@iki.fi>
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
# in spglib need to be used from Python. The reconstruction of the
# anharmonic interatomic constant set from the minimal set of
# constants is also implemented in this file for efficiency.

from libc.stdlib cimport malloc,free,div,div_t
from libc.math cimport round,fabs,sqrt

import copy

import numpy as np
import scipy as sp
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg

cimport cython
cimport numpy as np
np.import_array()
cimport cthirdorder_core

# NOTE: all indices used in this module are zero-based.

# Maximum matrix size (rows*cols) for the dense method.
DEF MAXDENSE=33554432

# Permutations of 3 elements listed in the same order as in the old
# Fortran code.
cdef int[:,:] permutations=np.array([
    [0,1,2],
    [1,0,2],
    [2,1,0],
    [0,2,1],
    [1,2,0],
    [2,0,1]],dtype=np.intc)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline int _ind2id(int[:] icell,int ispecies,int[:] ngrid,int nspecies):
    """
    Merge a set of cell+atom indices into a single index into a supercell.
    """
    return (icell[0]+(icell[1]+icell[2]*ngrid[1])*ngrid[0])*nspecies+ispecies


@cython.boundscheck(False)
@cython.wraparound(False)
cdef tuple _id2ind(int[:] ngrid,int nspecies):
    """
    Create a map between supercell indices to cell+atom indices.
    """
    cdef div_t tmp
    cdef int ii,ntot
    cdef int[:,:] icell
    cdef int[:] ispecies

    cdef np.ndarray np_icell,np_ispecies

    ntot=ngrid[0]*ngrid[1]*ngrid[2]*nspecies
    np_icell=np.empty((3,ntot),dtype=np.intc)
    np_ispecies=np.empty(ntot,dtype=np.intc)
    icell=np_icell
    ispecies=np_ispecies
    for ii in xrange(ntot):
        tmp=div(ii,nspecies)
        ispecies[ii]=tmp.rem
        tmp=div(tmp.quot,ngrid[0])
        icell[0,ii]=tmp.rem
        tmp=div(tmp.quot,ngrid[1])
        icell[1,ii]=tmp.rem
        icell[2,ii]=tmp.quot
    return (np_icell,np_ispecies)


# Thin, specialized wrapper around spglib.
cdef class SymmetryOperations:
  """
  Object that contains all the interesting information about the
  crystal symmetry group of a set of atoms.
  """
  cdef double[:,:] __lattvectors
  cdef int[:] __types
  cdef double[:,:] __positions
  cdef readonly str symbol
  cdef double[:] __shift
  cdef double[:,:] __transform
  cdef double[:,:,:] __rotations
  cdef double[:,:,:] __crotations
  cdef double[:,:] __translations
  cdef double[:,:] __ctranslations
  cdef double[:] __norms
  cdef double c_latvectors[3][3]
  cdef int *c_types
  cdef double (*c_positions)[3]
  cdef readonly int natoms,nsyms
  cdef readonly double symprec

  property lattice_vectors:
      def __get__(self):
          return np.asarray(self.__lattvectors)
  property types:
      def __get__(self):
          return np.asarray(self.__lattvectors)
  property positions:
      def __get__(self):
          return np.asarray(self.__positions)
  property origin_shift:
      def __get__(self):
          return np.asarray(self.__shift)
  property transformation_matrix:
      def __get__(self):
          return np.asarray(self.__transform)
  property rotations:
      def __get__(self):
          return np.asarray(self.__rotations)
  property translations:
      def __get__(self):
          return np.asarray(self.__translations)
  property crotations:
      def __get__(self):
          return np.asarray(self.__crotations)
  property ctranslations:
      def __get__(self):
          return np.asarray(self.__ctranslations)

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
      for i in xrange(3):
          for j in xrange(3):
              self.c_latvectors[i][j]=self.__lattvectors[i,j]
      for i in xrange(self.natoms):
          self.c_types[i]=self.__types[i]
          for j in xrange(3):
              self.c_positions[i][j]=self.__positions[i,j]

  cdef void __spg_get_dataset(self) except *:
      """
      Thin, slightly selective wrapper around spg_get_dataset(). The
      interesting information is copied out to Python objects and the
      rest discarded.
      """
      cdef int i,j,k
      cdef double[:] tmp1d
      cdef double[:,:] lat,tmp2d
      cdef double[:,:,:] rot
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
      self.__shift=np.empty((3,),dtype=np.double)
      self.__transform=np.empty((3,3),dtype=np.double)
      self.nsyms=data.n_operations
      self.__rotations=np.empty((self.nsyms,3,3),
                                   dtype=np.double)
      self.__translations=np.empty((self.nsyms,3),
                                      dtype=np.double)
      for i in xrange(3):
          self.__shift[i]=data.origin_shift[i]
          for j in xrange(3):
              self.__transform[i,j]=data.transformation_matrix[i][j]
      for i in xrange(self.nsyms):
          for j in xrange(3):
              self.__translations[i,j]=data.translations[i][j]
              for k in xrange(3):
                  self.__rotations[i,j,k]=data.rotations[i][j][k]
      lat=np.transpose(self.__lattvectors)
      rot=np.transpose(self.__rotations,(0,2,1))
      self.__crotations=np.empty_like(self.__rotations)
      self.__ctranslations=np.empty_like(self.__translations)
      for i in xrange(self.nsyms):
          tmp2d=np.dot(sp.linalg.solve(lat,rot[i,:,:]),lat)
          self.__crotations[i,:,:]=tmp2d.T
          tmp1d=np.dot(lat,self.__translations[i,:])
          self.__ctranslations[i,:]=tmp1d
      cthirdorder_core.spg_free_dataset(data)

  def __cinit__(self,lattvectors,types,positions,symprec=1e-5):
      cdef int i
      self.__lattvectors=np.array(lattvectors,dtype=np.double)
      self.__types=np.array(types,dtype=np.intc)
      self.__positions=np.array(positions,dtype=np.double)
      self.__norms=np.empty((3,),dtype=np.double)
      for i in xrange(3):
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

  cdef __apply_all(self,double[:] r_in):
      """
      Apply all symmetry operations to a vector and return the results.
      """
      cdef int ii,jj,kk
      cdef np.ndarray r_out
      cdef double[:,:] vr_out

      r_out=np.zeros((3,self.nsyms),dtype=np.double)
      vr_out=r_out
      for ii in xrange(self.nsyms):
          for jj in xrange(3):
              for kk in xrange(3):
                  vr_out[jj,ii]+=self.__crotations[ii,jj,kk]*r_in[kk]
              vr_out[jj,ii]+=self.__ctranslations[ii,jj]
      return r_out

  cpdef map_supercell(self,dict sposcar):
      """
      Each symmetry operation defines an atomic permutation in a supercell. This method
      returns an array with those permutations. The supercell must be compatible with
      the unit cell used to create the object.
      """
      cdef int ntot
      cdef int i,ii,isym,ispecies
      cdef int[:] ngrid,ind_species,vec
      cdef int[:,:] ind_cell,v_nruter
      cdef double dmin
      cdef double[:] car,diffs
      cdef double[:,:] car_sym,positions,latvec,tmp
      cdef np.ndarray np_icell,np_ispecies,nruter

      positions=sposcar["positions"]
      latvec=sposcar["lattvec"]
      ngrid=np.array([sposcar["na"],sposcar["nb"],sposcar["nc"]],
                     dtype=np.intc)
      ntot=positions.shape[1]
      natoms=ntot//(ngrid[0]*ngrid[1]*ngrid[2])
      np_icell,np_ispecies=_id2ind(ngrid,natoms)
      ind_cell=np_icell
      ind_species=np_ispecies
      nruter=np.empty((self.nsyms,ntot),dtype=np.intc)
      car=np.empty(3)
      v_nruter=nruter
      vec=np.empty(3,dtype=np.intc)
      diffs=np.zeros(natoms,dtype=np.double)
      for i in xrange(ntot):
          for ii in xrange(3):
              car[ii]=(positions[0,i]*latvec[ii,0]+
                       positions[1,i]*latvec[ii,1]+
                       positions[2,i]*latvec[ii,2])
          car_sym=self.__apply_all(car)
          tmp=np.mod(sp.linalg.solve(latvec,car_sym),1.)
          for ii in xrange(3):
              for isym in xrange(self.nsyms):
                  tmp[ii,isym]*=ngrid[ii]
          for isym in xrange(self.nsyms):
              for ii in xrange(3):
                  vec[ii]=int(round(tmp[ii,isym]))
                  tmp[ii,isym]-=vec[ii]
              for ii in xrange(natoms):
                  diffs[ii]=(fabs(tmp[0,isym]-self.__positions[ii,0])+
                             fabs(tmp[1,isym]-self.__positions[ii,1])+
                             fabs(tmp[2,isym]-self.__positions[ii,2]))
              ispecies=0
              dmin=diffs[0]
              for ii in xrange(1,natoms):
                  if diffs[ii]<dmin:
                      ispecies=ii
                      dmin=diffs[ii]
              v_nruter[isym,i]=_ind2id(vec,ispecies,ngrid,natoms)
      return nruter

@cython.boundscheck(False)
def reconstruct_ifcs(phipart,wedgeres,list4,poscar,sposcar):
    """
    Recover the full anharmonic IFC set from the irreducible set of
    force constants and the information obtained from wedge().
    """
    cdef int ii,jj,ll,mm,nn,kk,ss,tt,ix
    cdef int nlist,nnonzero,natoms,ntot,tribasisindex,colindex,nrows,ncols
    cdef int[:] naccumindependent
    cdef int[:,:,:] vind1
    cdef int[:,:,:] vind2
    cdef int[:,:,:] vequilist
    cdef double[:] aphilist
    cdef double[:,:] vaa
    cdef double[:,:,:] vphipart
    cdef double[:,:,:,:] doubletrans
    cdef double[:,:,:,:,:,:] vnruter

    nlist=wedgeres["Nlist"]
    natoms=len(poscar["types"])
    ntot=len(sposcar["types"])
    vnruter=np.zeros((3,3,3,natoms,ntot,ntot))
    naccumindependent=np.insert(np.cumsum(wedgeres["NIndependentBasis"],
                                                dtype=np.intc),0,[0])
    ntotalindependent=naccumindependent[-1]
    vphipart=phipart
    for ii,e in enumerate(list4):
        vnruter[e[2],e[3],:,e[0],e[1],:]=vphipart[:,ii,:]
    philist=[]
    for ii in xrange(nlist):
        for jj in xrange(wedgeres["NIndependentBasis"][ii]):
            ll=wedgeres["IndependentBasis"][jj,ii]//9
            mm=(wedgeres["IndependentBasis"][jj,ii]%9)//3
            nn=wedgeres["IndependentBasis"][jj,ii]%3
            philist.append(vnruter[ll,mm,nn,
                                  wedgeres["List"][0,ii],
                                  wedgeres["List"][1,ii],
                                  wedgeres["List"][2,ii]])
    aphilist=np.array(philist)
    vind1=-np.ones((natoms,ntot,ntot),dtype=np.intc)
    vind2=-np.ones((natoms,ntot,ntot),dtype=np.intc)
    vequilist=wedgeres["ALLEquiList"]
    for ii in xrange(nlist):
        for jj in xrange(wedgeres["Nequi"][ii]):
            vind1[vequilist[0,jj,ii],
                  vequilist[1,jj,ii],
                  vequilist[2,jj,ii]]=ii
            vind2[vequilist[0,jj,ii],
                  vequilist[1,jj,ii],
                  vequilist[2,jj,ii]]=jj

    vtrans=wedgeres["TransformationArray"]

    nrows=ntotalindependent
    ncols=natoms*ntot*27

    if nrows*ncols<=MAXDENSE:
        print "- Storing the coefficients in a dense matrix"
        aa=np.zeros((nrows,ncols))
        vaa=aa
        colindex=0
        for ii in xrange(natoms):
            for jj in xrange(ntot):
                tribasisindex=0
                for ll in xrange(3):
                    for mm in xrange(3):
                        for nn in xrange(3):
                            for kk in xrange(ntot):
                                for ix in xrange(nlist):
                                    if vind1[ii,jj,kk]==ix:
                                        for ss in xrange(naccumindependent[ix],
                                                         naccumindependent[ix+1]):
                                            tt=ss-naccumindependent[ix]
                                            vaa[ss,colindex]+=vtrans[tribasisindex,tt,
                                                                     vind2[ii,jj,kk],ix]
                            tribasisindex+=1
                            colindex+=1
    else:
        print "- Storing the coefficients in a sparse matrix"
        i=[]
        j=[]
        v=[]
        colindex=0
        for ii in xrange(natoms):
            for jj in xrange(ntot):
                tribasisindex=0
                for ll in xrange(3):
                    for mm in xrange(3):
                        for nn in xrange(3):
                            for kk in xrange(ntot):
                                for ix in xrange(nlist):
                                    if vind1[ii,jj,kk]==ix:
                                        for ss in xrange(naccumindependent[ix],
                                                         naccumindependent[ix+1]):
                                            tt=ss-naccumindependent[ix]
                                            i.append(ss)
                                            j.append(colindex)
                                            v.append(vtrans[tribasisindex,tt,
                                                            vind2[ii,jj,kk],ix])
                            tribasisindex+=1
                            colindex+=1
        print "- \t Density: {0:.2g}%".format(100.*len(i)/float(nrows*ncols))
        aa=sp.sparse.coo_matrix((v,(i,j)),(nrows,ncols)).tocsr()
    D=sp.sparse.spdiags(aphilist,[0,],aphilist.size,aphilist.size,
                           format="csr")
    bbs=D.dot(aa)
    ones=np.ones_like(aphilist)
    multiplier=-sp.sparse.linalg.lsqr(bbs,ones)[0]
    compensation=D.dot(bbs.dot(multiplier))

    aphilist+=compensation

    # Build the final, full set of anharmonic IFCs.
    vnruter[:,:,:,:,:,:]=0.
    for ii in xrange(nlist):
        for jj in xrange(wedgeres["Nequi"][ii]):
            for ll in xrange(3):
                for mm in xrange(3):
                    for nn in xrange(3):
                        tribasisindex=(ll*3+mm)*3+nn
                        for ix in xrange(wedgeres["NIndependentBasis"][ii]):
                            vnruter[ll,mm,nn,vequilist[0,jj,ii],
                                    vequilist[1,jj,ii],
                                    vequilist[2,jj,ii]
                                    ]+=wedgeres["TransformationArray"][
                                        tribasisindex,ix,jj,ii]*aphilist[
                                            naccumindependent[ii]+ix]
    return vnruter


import sys

#### Experimental section: Cython replacements of old Fortran functions.
cdef class Wedge:
    """
    Objects of this class allow the user to extract irreducible sets
    of force constants and to reconstruct the full third-order IFC
    matrix from them.
    """
    cdef readonly SymmetryOperations symops
    cdef readonly dict poscar
    cdef readonly dict sposcar
    cdef readonly dict wedgeres
    cdef int[:,:] nequi
    cdef int[:,:,:] shifts
    cdef double[:,:] dmin
    cdef readonly double frange

    def __cinit__(self,poscar,sposcar,symops,dmin,nequi,shifts,frange):
        """
        Build the object by computing all the relevant information about
        irreducible IFCs.
        """
        self.poscar=poscar
        self.sposcar=sposcar
        self.symops=symops
        self.dmin=dmin
        self.nequi=nequi
        self.shifts=shifts
        self.frange=frange

        self._reduce()

    cdef _reduce(self):
        """
        C-level method that performs most of the actual work.
        """
        cdef int ngrid1,ngrid2,ngrid3,nsymm,natoms,ntot,summ
        cdef int ii,jj,kk,ll,iaux,jaux
        cdef int ibasis,jbasis,kbasis,ibasisprime,jbasisprime,kbasisprime
        cdef int ipermutation,iel
        cdef int indexijk,indexijkprime,indexrow
        cdef int[:] ngrid,ind_species,vec1,vec2,vec3
        cdef int[:,:] shifts27,shift2all,shift3all,
        cdef int[:,:] equilist,id_equi,ind_cell
        cdef double dist,frange2
        cdef double[:] car2,car3
        cdef double[:,:] latvec,coord,coordall,BB
        cdef double[:,:,:] orth
        cdef list llist,nequi,allequilist,alllist
        cdef list transformation,transformationaux,independentbasis

        frange2=self.frange*self.frange

        ngrid1=self.sposcar["na"]
        ngrid2=self.sposcar["nb"]
        ngrid3=self.sposcar["nc"]
        ngrid=np.array([ngrid1,ngrid2,ngrid3],dtype=np.intc)
        nsymm=self.symops.nsyms
        natoms=len(self.poscar["types"])
        ntot=len(self.sposcar["types"])
        vec1=np.empty(3,dtype=np.intc)
        vec2=np.empty(3,dtype=np.intc)
        vec3=np.empty(3,dtype=np.intc)

        latvec=self.sposcar["lattvec"]
        coord=np.dot(latvec,self.poscar["positions"])
        coordall=np.dot(latvec,self.sposcar["positions"])
        orth=np.transpose(self.symops.crotations,(1,2,0))
        car2=np.empty(3,dtype=np.double)
        car3=np.empty(3,dtype=np.double)

        summ=0
        llist=[]
        nequi=[]
        allequilist=[]
        transformation=[]
        transformationaux=[]
        independentbasis=[]
        alllist=[]

        iaux=0
        shifts27=np.empty((27,3),dtype=np.intc)
        for ii in xrange(-1,2):
            for jj in xrange(-1,2):
                for kk in xrange(-1,2):
                    shifts27[iaux,0]=ii
                    shifts27[iaux,1]=jj
                    shifts27[iaux,2]=kk
                    iaux+=1

        shift2all=np.empty((3,27),dtype=np.intc)
        shift3all=np.empty((3,27),dtype=np.intc)
        BB=np.empty((27,27),dtype=np.double)
        equilist=np.empty((3,nsymm*6),dtype=np.intc)
        id_equi=self.symops.map_supercell(self.sposcar)
        ind_cell,ind_species=_id2ind(ngrid,natoms)
        for ii in xrange(natoms):
            for jj in xrange(ntot):
                dist=self.dmin[ii,jj]
                if dist>=self.frange:
                    continue
                n2equi=self.nequi[ii,jj]
                for kk in xrange(n2equi):
                    shift2all[:,kk]=shifts27[self.shifts[ii,jj,kk],:]
                for kk in xrange(ntot):
                    dist=self.dmin[ii,kk]
                    if dist>=self.frange:
                        continue
                    n3equi=self.nequi[ii,kk]
                    for ll in xrange(n3equi):
                        shift3all[:,ll]=shifts27[self.shifts[ii,kk,ll],:]
                    d2_min=np.inf
                    for iaux in xrange(n2equi):
                        for ll in xrange(3):
                            car2[ll]=(shift2all[0,iaux]*latvec[ll,0]+
                                      shift2all[1,iaux]*latvec[ll,1]+
                                      shift2all[2,iaux]*latvec[ll,2]+
                                      coordall[ll,jj])
                        for jaux in xrange(n3equi):
                            for ll in xrange(3):
                                car3[ll]=(shift3all[0,jaux]*latvec[ll,0]+
                                          shift3all[1,jaux]*latvec[ll,1]+
                                          shift3all[2,jaux]*latvec[ll,2]+
                                          coordall[ll,kk])
                        d2_min=min(d2_min,
                                   (car3[0]-car2[0])**2+
                                   (car3[1]-car2[1])**2+
                                   (car3[2]-car2[2])**2)
                    if d2_min>=frange2:
                        continue
                    summ+=1
                    triplet=[ii,jj,kk]
                    if triplet in alllist:
                        continue
                    llist.append(triplet)
                    nequi.append(0)
                    allequilist.append([])
                    coeffi=np.zeros((6*nsymm*27,27),dtype=np.double)
                    nnonzero=0
                    transformation.append([])
                    for ipermutation in xrange(6):
                        triplet_permutation=[triplet[iel]
                                             for iel in permutations[ipermutation,:]]
                        for isym in xrange(nsymm):
                            triplet_sym=[id_equi[isym,triplet_permutation[0]],
                                         id_equi[isym,triplet_permutation[1]],
                                         id_equi[isym,triplet_permutation[2]]]
                            for ll in xrange(3):
                                vec1[ll]=ind_cell[ll,id_equi[isym,triplet_permutation[0]]]
                                vec2[ll]=ind_cell[ll,id_equi[isym,triplet_permutation[1]]]
                                vec3[ll]=ind_cell[ll,id_equi[isym,triplet_permutation[2]]]
                            ispecies1=ind_species[id_equi[isym,triplet_permutation[0]]]
                            ispecies2=ind_species[id_equi[isym,triplet_permutation[1]]]
                            ispecies3=ind_species[id_equi[isym,triplet_permutation[2]]]
                            for ll in xrange(3):
                                vec2[ll]=(vec2[ll]-vec1[ll])%ngrid[ll]
                                vec3[ll]=(vec3[ll]-vec1[ll])%ngrid[ll]
                            if not vec1[0]==vec1[1]==vec1[2]:
                                for ll in xrange(3):
                                    vec1[ll]=0
                                triplet_sym[0]=_ind2id(vec1,ispecies1,ngrid,natoms)
                                triplet_sym[1]=_ind2id(vec2,ispecies2,ngrid,natoms)
                                triplet_sym[2]=_ind2id(vec3,ispecies3,ngrid,natoms)
                            for ibasisprime in xrange(3):
                                for jbasisprime in xrange(3):
                                    for kbasisprime in xrange(3):
                                        indexijkprime=ibasisprime*9+jbasisprime*3+kbasisprime
                                        indexrow=ipermutation*nsymm*27+isym*27+indexijkprime
                                        for ibasis in xrange(3):
                                            for jbasis in xrange(3):
                                                for kbasis in xrange(3):
                                                    indexijk=ibasis*9+jbasis*3+kbasis
                                                    ibasispermut,jbasispermut,kbasispermut=[
                                                        [ibasis,jbasis,kbasis][iel] for
                                                        iel in permutations[ipermutation,:]]
                                                    BB[indexijkprime,indexijk]=(
                                                        orth[ibasisprime,ibasispermut,isym]*
                                                        orth[jbasisprime,jbasispermut,isym]*
                                                        orth[kbasisprime,kbasispermut,isym])
                            iaux=1
                            if not (ipermutation==0 and isym==0):
                                for ll in xrange(nequi[-1]):
                                    if triplet_sym==list(equilist[:,ll]):
                                        iaux=0
                            if iaux==1 and ((ipermutation==0 and isym==0) or triplet_sym!=triplet):
                                nequi[-1]+=1
                                for ll in xrange(3):
                                    equilist[ll,nequi[-1]-1]=triplet_sym[ll]
                                allequilist[-1].append(triplet_sym)
                                alllist.append(triplet_sym)
                                transformation[-1].append(np.array(BB))
                            if triplet_sym==triplet:
                                for indexijkprime in xrange(27):
                                    nonzero=False
                                    for indexijk in xrange(27):
                                        if indexijkprime==indexijk:
                                            BB[indexijkprime,indexijk]-=1.
                                        if abs(BB[indexijkprime,indexijk])>1e-12:
                                            nonzero=True
                                        else:
                                            BB[indexijkprime,indexijk]=0.
                                    if nonzero:
                                        coeffi[nnonzero,:]=BB[indexijkprime,:]
                                        nnonzero+=1
                    coeffi_reduced=np.zeros((max(nnonzero,27),27),dtype=np.double)
                    coeffi_reduced[:nnonzero,:]=coeffi[:nnonzero,:]
                    b,independent=gaussian(coeffi_reduced)
                    transformationaux.append(b)
                    independentbasis.append(independent)
        transformationarray=np.zeros((27,27,nsymm*6,len(llist)),dtype=np.double)
        for ii in xrange(len(llist)):
            for jj in xrange(nequi[ii]):
                transformationarray[:,:len(independentbasis[ii]),jj,ii]=np.dot(
                    transformation[ii][jj][:,:],
                    transformationaux[ii][:,:len(independentbasis[ii])])
                for kk in xrange(27):
                    for ll in xrange(27):
                        if abs(transformationarray[kk,ll,jj,ii])<1e-12:
                            transformationarray[kk,ll,jj,ii]=0.
        # TODO: examine carefully
        print llist
        nlist=len(llist)
        self.wedgeres=dict()
        self.wedgeres["Nlist"]=nlist
        self.wedgeres["Nequi"]=np.array(nequi)
        self.wedgeres["List"]=np.array(llist).T
        self.wedgeres["NIndependentBasis"]=np.array([len(i) for i in independentbasis])
        self.wedgeres["ALLEquiList"]=np.empty((nlist,nsymm*6,3),dtype=np.intc)
        self.wedgeres["IndependentBasis"]=np.empty((nlist,27),dtype=np.intc)
        for i in xrange(nlist):
            for j in xrange(nequi[i]):
                self.wedgeres["ALLEquiList"][i,j,:]=allequilist[i][j]
            self.wedgeres["IndependentBasis"][i,:len(independentbasis[i])]=independentbasis[i]
        self.wedgeres["IndependentBasis"]=self.wedgeres["IndependentBasis"].T
        self.wedgeres["TransformationArray"]=transformationarray
        self.wedgeres["ALLEquiList"]=np.transpose(self.wedgeres["ALLEquiList"],(2,1,0))


DEF EPS=1e-10
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef tuple gaussian(double[:,:] a):
    """
    Specialized version of Gaussian elimination.
    """
    cdef int i,j,k,irow
    cdef int row,column,ndependent,nindependent
    cdef double tmp
    cdef int[:] dependent,independent

    row=a.shape[0]
    col=a.shape[1]

    dependent=np.empty(col,dtype=np.intc)
    independent=np.empty(col,dtype=np.intc)
    b=np.zeros((col,col))

    irow=0
    ndependent=0
    nindependent=0
    for k in xrange(min(row,col)):
        for i in xrange(row):
            if fabs(a[i,k])<EPS:
                a[i,k]=0.
        for i in xrange(irow+1,row):
            if fabs(a[i,k])-fabs(a[irow,k])>EPS:
                for j in xrange(k,col):
                    tmp=a[irow,j]
                    a[irow,j]=a[i,j]
                    a[i,j]=tmp
        if fabs(a[irow,k])>EPS:
            dependent[ndependent]=k
            ndependent+=1
            for j in xrange(col-1,k,-1):
                a[irow,j]/=a[irow,k]
            a[irow,k]=1.
            for i in xrange(row):
                if i==irow:
                    continue
                for j in xrange(col-1,k,-1):
                    a[i,j]-=a[i,k]*a[irow,j]/a[irow,k]
                a[i,k]=0.
            if irow<row-1:
                irow+=1
        else:
            independent[nindependent]=k
            nindependent+=1
    for j in xrange(nindependent):
        for i in xrange(ndependent):
            b[dependent[i],j]=-a[i,independent[j]]
        b[independent[j],j]=1.
    return (b,list(independent[:nindependent]))
