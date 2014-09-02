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

from libc.stdlib cimport malloc,free
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
        [2,0,1]],dtype=np.int32)


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
      self.__shift=np.empty((3,),dtype=np.float64)
      self.__transform=np.empty((3,3),dtype=np.float64)
      self.nsyms=data.n_operations
      self.__rotations=np.empty((self.nsyms,3,3),
                                   dtype=np.float64)
      self.__translations=np.empty((self.nsyms,3),
                                      dtype=np.float64)
      for i in xrange(3):
          self.__shift[i]=data.origin_shift[i]
          for j in xrange(3):
              self.__transform[i,j]=data.transformation_matrix[i][j]
      for i in xrange(self.nsyms):
          for j in xrange(3):
              self.__translations[i,j]=data.translations[i][j]
              for k in xrange(3):
                  self.__rotations[i,j,k]=data.rotations[i][j][k]
      cthirdorder_core.spg_free_dataset(data)

  def __cinit__(self,lattvectors,types,positions,symprec=1e-5):
      cdef int i
      self.__lattvectors=np.array(lattvectors,dtype=np.float64)
      self.__types=np.array(types,dtype=np.int32)
      self.__positions=np.array(positions,dtype=np.float64)
      self.__norms=np.empty((3,),dtype=np.float64)
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
                                                dtype=np.int32),0,[0])
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
    vind1=-np.ones((natoms,ntot,ntot),dtype=np.int32)
    vind2=-np.ones((natoms,ntot,ntot),dtype=np.int32)
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
        print "- Using a dense QR factorization algorithm"
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
        Q,R,P=sp.linalg.qr(aa,mode="economic",pivoting=True)
        nnonzero=(np.abs(np.diag(R))>=1e-12).sum()
        bb=np.array(Q[:,:nnonzero])
        D=np.diag(aphilist)
        ones=np.ones_like(aphilist)
        bb=np.dot(D,bb)
        multiplier=-sp.linalg.lstsq(bb,ones)[0]
        compensation=np.dot(D,np.dot(bb,multiplier))
    else:
        print "- Using a sparse least-squares method"
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
def nofortran_pywedge(poscar,sposcar,symops,frange):
    """
    Wrapper around nofortran_wedge() that returns a python dictionary with all
    relevant information about the irreducible displacements.
    """
    crotations=np.empty_like(symops.rotations)

    ngrid1=sposcar["na"]
    ngrid2=sposcar["nb"]
    ngrid3=sposcar["nc"]
    nsymm=symops.translations.shape[0]
    natoms=len(poscar["types"])
    ntot=len(sposcar["types"])

    for i in xrange(nsymm):
        crotations[i,:,:]=np.dot(
            sp.linalg.solve(poscar["lattvec"].T,symops.rotations[i,:,:].T),
            poscar["lattvec"].T).T
    coord=np.dot(poscar["lattvec"],poscar["positions"])
    coordall=np.dot(sposcar["lattvec"],sposcar["positions"])
    latvec=poscar["lattvec"]
    invlatvec=sp.linalg.inv(latvec)
    trans=symops.translations.T
    orth=np.transpose(crotations,(1,2,0))
    nequi,llist,allequilist,transformationarray,independentbasis=nofortran_wedge(
        latvec,invlatvec,coord,coordall,orth,trans,natoms,ngrid1,ngrid2,ngrid3,nsymm,frange)
    nlist=len(llist)
    nruter=dict()
    nruter["Nlist"]=nlist
    nruter["Nequi"]=np.array(nequi)
    nruter["List"]=np.array(llist).T
    nruter["NIndependentBasis"]=np.array([len(i) for i in independentbasis])
    nruter["ALLEquiList"]=np.empty((nlist,nsymm*6,3),dtype=np.int32)
    nruter["IndependentBasis"]=np.empty((nlist,27),dtype=np.int32)
    for i in xrange(nlist):
        for j in xrange(nequi[i]):
            nruter["ALLEquiList"][i,j,:]=allequilist[i][j]
        nruter["IndependentBasis"][i,:len(independentbasis[i])]=independentbasis[i]
    nruter["IndependentBasis"]=nruter["IndependentBasis"].T
    nruter["TransformationArray"]=transformationarray
    return nruter


def nofortran_wedge(latvec,invlatvec,coord,coordall,orth,trans,natoms,
                    ngrid1,ngrid2,ngrid3,nsymm,forcerange):
    """
    Determine a minimal set of third-order derivatives of the energy
    needed to obtain all anharmonic IFCs withing the cutoff radius
    ForceRange. The description of the constants is returned in cList;
    the rest of the output arguments are necessary for the
    reconstruction since they describe the equivalences and
    transformation rules between atomic triplets.
    """
    summ=0
    nlist=0
    llist=[]
    nequi=[]
    allequilist=[]
    transformation=[]
    transformationaux=[]
    independentbasis=[]
    alllist=[]

    shift2all=np.empty((3,27),dtype=np.int32)
    shift3all=np.empty((3,27),dtype=np.int32)
    BB=np.empty((27,27),dtype=np.float64)
    equilist=np.empty((3,nsymm*6),dtype=np.int32)

    id_equi=nofortran_symmetry(nsymm,natoms,latvec,invlatvec,coord,
                               ngrid1,ngrid2,ngrid3,orth,trans)
    ind_cell,ind_species=nofortran_id2ind(ngrid1,ngrid2,ngrid3,natoms)
    for ii in xrange(natoms):
        for jj in xrange(ngrid1*ngrid2*ngrid3*natoms):
            dist_min=np.inf
            n2equi=0
            for ix in xrange(-1,2):
                for iy in xrange(-1,2):
                    for iz in xrange(-1,2):
                        dist=sp.linalg.norm(
                            ix*ngrid1*latvec[:,0]+
                            iy*ngrid2*latvec[:,1]+
                            iz*ngrid3*latvec[:,2]+
                            coordall[:,jj]-coordall[:,ii])
                        if dist<dist_min:
                            dist_min=dist
            for ix in xrange(-1,2):
                for iy in xrange(-1,2):
                    for iz in xrange(-1,2):
                        dist=sp.linalg.norm(
                            ix*ngrid1*latvec[:,0]+
                            iy*ngrid2*latvec[:,1]+
                            iz*ngrid3*latvec[:,2]+
                            coordall[:,jj]-coordall[:,ii])
                        if abs(dist-dist_min)<1e-2:
                            shift2all[:,n2equi]=[ix,iy,iz]
                            n2equi+=1
            dist=dist_min
            if dist<forcerange:
                for kk in xrange(ngrid1*ngrid2*ngrid3*natoms):
                    dist_min=np.inf
                    n3equi=0
                    for ix in xrange(-1,2):
                        for iy in xrange(-1,2):
                            for iz in xrange(-1,2):
                                dist=sp.linalg.norm(
                                    ix*ngrid1*latvec[:,0]+
                                    iy*ngrid2*latvec[:,1]+
                                    iz*ngrid3*latvec[:,2]+
                                    coordall[:,kk]-coordall[:,ii])
                                if dist<dist_min:
                                    dist_min=dist
                    for ix in xrange(-1,2):
                        for iy in xrange(-1,2):
                            for iz in xrange(-1,2):
                                dist=sp.linalg.norm(
                                    ix*ngrid1*latvec[:,0]+
                                    iy*ngrid2*latvec[:,1]+
                                    iz*ngrid3*latvec[:,2]+
                                    coordall[:,kk]-coordall[:,ii])
                                if abs(dist-dist_min)<1e-2:
                                    shift3all[:,n3equi]=[ix,iy,iz]
                                    n3equi+=1
                    dist=dist_min
                    dist_min=np.inf
                    for iaux in xrange(n2equi):
                        car2=(shift2all[0,iaux]*ngrid1*latvec[:,0]+
                              shift2all[1,iaux]*ngrid2*latvec[:,1]+
                              shift2all[2,iaux]*ngrid3*latvec[:,2]+coordall[:,jj])
                        for jaux in xrange(n3equi):
                            car3=(shift3all[0,jaux]*ngrid1*latvec[:,0]+
                                  shift3all[1,jaux]*ngrid2*latvec[:,1]+
                                  shift3all[2,jaux]*ngrid3*latvec[:,2]+coordall[:,kk])
                        dist1=scipy.linalg.norm(car3-car2)
                        if dist1<dist_min:
                            dist_min=dist1
                            shift2=shift2all[:,iaux]
                            shift3=shift3all[:,jaux]
                    dist1=dist_min
                    if dist<forcerange and dist1<forcerange:
                        summ+=1
                        triplet=[ii,jj,kk]
                        if not triplet in alllist:
                            llist.append(triplet)
                            nequi.append(0)
                            allequilist.append([])
                            coeffi=np.zeros((6*nsymm*27,27),dtype=np.float64)
                            nnonzero=0
                            transformation.append([])
                            for ipermutation in xrange(6):
                                triplet_permutation=[
                                    triplet[iel] for iel in permutations[ipermutation,:]
                                    ]
                                for isym in xrange(nsymm):
                                    triplet_sym=[id_equi[isym,triplet_permutation[0]],
                                                 id_equi[isym,triplet_permutation[1]],
                                                 id_equi[isym,triplet_permutation[2]]]
                                    vec1=ind_cell[:,id_equi[isym,triplet_permutation[0]]]
                                    ispecies1=ind_species[id_equi[isym,triplet_permutation[0]]]
                                    vec2=ind_cell[:,id_equi[isym,triplet_permutation[1]]]
                                    ispecies2=ind_species[id_equi[isym,triplet_permutation[1]]]
                                    vec3=ind_cell[:,id_equi[isym,triplet_permutation[2]]]
                                    ispecies3=ind_species[id_equi[isym,triplet_permutation[2]]]
                                    if not np.all(vec1==0):
                                        triplet_sym[0]=nofortran_ind2id(np.zeros(3),
                                                                        ispecies1,ngrid1,ngrid2,natoms)
                                        triplet_sym[1]=nofortran_ind2id((vec2-vec1)%(ngrid1,ngrid2,ngrid3),
                                                                        ispecies2,ngrid1,ngrid2,natoms)
                                        triplet_sym[2]=nofortran_ind2id((vec3-vec1)%(ngrid1,ngrid2,ngrid3),
                                                                        ispecies3,ngrid1,ngrid2,natoms)
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
                                                                iel in permutations[ipermutation,:]
                                                                ]
                                                            BB[indexijkprime,indexijk]=(
                                                                orth[ibasisprime,ibasispermut,isym]*
                                                                orth[jbasisprime,jbasispermut,isym]*
                                                                orth[kbasisprime,kbasispermut,isym])
                                    iaux=1
                                    if not (ipermutation==0 and isym==0):
                                        for ll in xrange(nequi[-1]):
                                            if triplet_sym==list(equilist[:,ll]):
                                                iaux=0
                                    if iaux==1:
                                        if (ipermutation==0 and isym==0) or triplet_sym!=triplet:
                                            nequi[-1]+=1
                                            equilist[:,nequi[-1]-1]=triplet_sym
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
                            coeffi_reduced=np.zeros((max(nnonzero,27),27),dtype=np.float64)
                            coeffi_reduced[:nnonzero,:]=coeffi[:nnonzero,:]
                            b,independent=nofortran_gaussian(coeffi_reduced)
                            transformationaux.append(b)
                            independentbasis.append(independent)
    transformationarray=np.zeros((27,27,nsymm*6,len(llist)),dtype=np.float64)
    for ii in xrange(len(llist)):
        for jj in xrange(nequi[ii]):
            transformationarray[:,:len(independentbasis[ii]),jj,ii]=np.dot(
                transformation[ii][jj][:,:],
                transformationaux[ii][:,:len(independentbasis[ii])])
            for kk in xrange(27):
                for ll in xrange(27):
                    if abs(transformationarray[kk,ll,jj,ii])<1e-12:
                        transformationarray[kk,ll,jj,ii]=0.
    return (nequi,llist,allequilist,transformationarray,independentbasis)



def nofortran_symmetry(nsymm,natoms,latvec,invlatvec,coord,
                       ngrid1,ngrid2,ngrid3,orth,trans):
    """
    Each symmetry operation defines a mapping between atom indices in
    the supercell. This subroutine fills a matrix with those
    permutations.
    """
    ind_cell,ind_species=nofortran_id2ind(ngrid1,ngrid2,ngrid3,natoms)
    nruter=np.empty((nsymm,natoms*ngrid1*ngrid2*ngrid3),dtype=np.int32)
    for i in xrange(natoms*ngrid1*ngrid2*ngrid3):
        vec=ind_cell[:,i]
        ispecies=ind_species[i]
        car=nofortran_lattice2car(latvec,coord,vec,ispecies)
        car_sym=nofortran_symm(nsymm,latvec,car,orth,trans)
        for isym in xrange(nsymm):
            vec,ispecies_sym=nofortran_car2lattice(
                natoms,latvec,invlatvec,coord,car_sym[:,isym])
            vec[0]=vec[0]%ngrid1
            vec[1]=vec[1]%ngrid2
            vec[2]=vec[2]%ngrid3
            nruter[isym,i]=nofortran_ind2id(vec,ispecies_sym,ngrid1,ngrid2,natoms)
    return nruter


def nofortran_symm(nsymm,latvec,r_in,orth,trans):
    """
    Apply a symmetry operation to a vector and return the result.
    """
    r_out=np.empty((3,nsymm))
    for ii in xrange(nsymm):
        disp=trans[0,ii]*latvec[:,0]+trans[1,ii]*latvec[:,1]+trans[2,ii]*latvec[:,2]
        r_out[:,ii]=np.dot(orth[:,:,ii],r_in)+disp
    return r_out


def nofortran_car2lattice(natoms,latvec,invlatvec,coord,car):
    """
    Return the unit cell and atom indices of an element of the
    supercell based on its Cartesian coordinates.
    """
    tmp2=np.empty((3,natoms))
    for i in xrange(natoms):
        tmp2[:,i]=car-coord[:,i]
    tmp2=np.dot(invlatvec,tmp2)
    for i in xrange(natoms):
        icell=np.round(tmp2[:,i]).astype(np.int32)
        displ=icell[0]*latvec[:,0]+icell[1]*latvec[:,1]+icell[2]*latvec[:,2]-(car-coord[:,i])
        dist=np.dot(displ,displ)
        if dist<1e-4:
            iatom=i
            break
    return (icell,iatom)


def nofortran_lattice2car(latvec,coord,icell,iatom):
    """
    Inverse of the previous subroutine: converts from atom+cell
    indices to Cartesian coordinates.
    """
    return icell[0]*latvec[:,0]+icell[1]*latvec[:,1]+icell[2]*latvec[:,2]+coord[:,iatom]


def nofortran_id2ind(ngrid1,ngrid2,ngrid3,nspecies):
    """
    Generate a mapping between unit cell+atom indices to atom indices
    in the supercell.
    """
    icell=np.empty((3,ngrid1*ngrid2*ngrid3*nspecies),dtype=np.int32)
    ispecies=np.empty(ngrid1*ngrid2*ngrid3*nspecies,dtype=np.int32)
    for ii in xrange(ngrid1*ngrid2*ngrid3*nspecies):
        ispecies[ii]=ii%nspecies
        tmp=ii//nspecies
        icell[2,ii]=tmp//(ngrid1*ngrid2)
        icell[1,ii]=(tmp%(ngrid1*ngrid2))//ngrid1
        icell[0,ii]=tmp%ngrid1
    return (icell,ispecies)


def nofortran_ind2id(icell,ispecies,ngrid1,ngrid2,nspecies):
    """
    Merge a set of cell+atom indices into a single index into the supercell.
    """
    return (icell[0]+(icell[1]+icell[2]*ngrid2)*ngrid1)*nspecies+ispecies


DEF EPS=1e-10
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.wraparound(True)
cdef nofortran_gaussian(double[:,:] a):
    """
    Specialized version of Gaussian elimination.
    """
    cdef int i,j,k,irow
    cdef int row,column,ndependent,nindependent
    cdef double tmp
    cdef int[:] dependent,independent

    row=a.shape[0]
    col=a.shape[1]

    dependent=np.empty(col,dtype=np.int32)
    independent=np.empty(col,dtype=np.int32)
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
