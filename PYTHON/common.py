#!/usr/bin/python
# -*- coding: utf-8 -*-
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

import sys
import os
import copy
import itertools
import collections
import contextlib
import numpy
import numpy.linalg
import scipy
import scipy.linalg


@contextlib.contextmanager
def dir_context(directory):
    """
    Context manager used to run code in another directory.
    """
    curdir=os.getcwd()
    os.chdir(directory)
    try:
        yield directory
    finally:
        os.chdir(curdir)


def read_POSCAR(directory):
    """
    Return all the relevant information contained in a POSCAR file.
    """
    with dir_context(directory):
        nruter=dict()
        nruter["lattvec"]=numpy.empty((3,3))
        f=open(os.path.join(directory,"POSCAR"),"r")
        firstline=f.next()
        factor=.1*float(f.next().strip())
        for i in range(3):
            nruter["lattvec"][:,i]=[float(j) for j in f.next().split()]
        nruter["lattvec"]*=factor
        line=f.next()
        fields=f.next().split()
        old=False
        try:
            int(fields[0])
        except ValueError:
            old=True
        if old:
            nruter["elements"]=firstline.split()
            nruter["numbers"]=numpy.array([int(i) for i in line.split()])
            typeline="".join(fields)
        else:
            nruter["elements"]=line.split()
            nruter["numbers"]=numpy.array([int(i) for i in fields])
            typeline=f.next()
        natoms=nruter["numbers"].sum()
        nruter["positions"]=numpy.empty((3,natoms))
        for i in range(natoms):
            nruter["positions"][:,i]=[float(j) for j in f.next().split()]
        f.close()
    nruter["types"]=[]
    for i in range(len(nruter["numbers"])):
        nruter["types"]+=[i]*nruter["numbers"][i]
    if typeline[0]=="C":
        nruter["positions"][:,i]=scipy.linalg.solve(nruter["lattvec"],
                                                    nruter["positions"])
    return nruter


def gen_SPOSCAR(poscar,na,nb,nc):
    """
    Create a dictionary similar to the first argument but describing a
    supercell.
    """
    nruter=dict()
    nruter["lattvec"]=numpy.array(poscar["lattvec"])
    nruter["lattvec"][:,0]*=na
    nruter["lattvec"][:,1]*=nb
    nruter["lattvec"][:,2]*=nc
    nruter["elements"]=copy.copy(poscar["elements"])
    nruter["numbers"]=na*nb*nc*poscar["numbers"]
    nruter["positions"]=numpy.empty((3,poscar["positions"].shape[1]*na*nb*nc))
    pos=0
    for pos,(k,j,i,iat) in enumerate(itertools.product(range(nc),
                                                       range(nb),
                                                       range(na),
                                                       range(
                                    poscar["positions"].shape[1]))):
        nruter["positions"][:,pos]=(poscar["positions"][:,iat]+[i,j,k])/[
            na,nb,nc]
    nruter["types"]=[]
    for i in range(len(nruter["numbers"])):
        nruter["types"]+=[i]*nruter["numbers"][i]
    return nruter


def calc_frange2(poscar):
    """
    Return the squared maximum distance between fourth neighbors in
    the structure.
    """
    nat=poscar["positions"].shape[1]
    tensor=numpy.dot(poscar["lattvec"].T,poscar["lattvec"])
    calc_norm2=lambda x:numpy.dot(x,numpy.dot(tensor,x))
    calc_dist2=lambda x,y:calc_norm2(x-y)
    d2=numpy.empty((nat,nat))
    for i in range(nat-1):
        d2[i,i]=0.
        posi=poscar["positions"][:,i]
        for j in range(i+1,nat):
            d2min=numpy.inf
            for (ja,jb,jc) in itertools.product(range(-1,2),
                                                range(-1,2),
                                                range(-1,2)):
                posj=poscar["positions"][:,j]+[ja,jb,jc]
                d2new=calc_dist2(posi,posj)
                if d2new<d2min:
                    d2min=d2new
            d2[j,i]=d2[i,j]=d2min
    tofourth=[]
    for i in range(nat):
        ds=d2[i,:].tolist()
        ds.sort()
        u=[]
        for j in ds:
            for k in u:
                if numpy.allclose(k,j):
                    break
            else:
                u.append(j)
        try:
            tofourth.append(.5*(u[3]+u[4]))
        except IndexError:
            tofourth.append(1.1*max(u))
    return max(tofourth)


def symmetry_map(r_in,symops,na,nb,nc):
    """
    Return all the images of a vector (in supercell coordinates) through
    the symmetry operations of the system.
    """
    s2u=numpy.diag([na,nb,nc]).astype(numpy.float64)
    u2s=numpy.diag([1./na,1./nb,1./nc])
    nops=symops.translations.shape[0]
    r_out=numpy.empty((3,nops))
    for i in range(nops):
        r_out[:,i]=reduce(numpy.dot,
                          [u2s,symops.rotations[i,:,:],s2u,r_in])
        r_out[:,i]+=numpy.dot(u2s,symops.translations[i,:])
    return r_out


def ind2id(i,j,k,iat,na,nb,nc,natom):
    """
    Map four indices to a single atom id.
    """
    return iat+natom*(i+na*(j+nb*k))


def id2ind(anid,na,nb,nc,nat):
    """
    Split an atom id into four indices.
    """
    tmp,iat=divmod(anid,nat)
    tmp,i=divmod(tmp,na)
    k,j=divmod(tmp,nb)
    return dict(i=i,j=j,k=k,iat=iat)


def gen_equivalences(sposcar,symops,na,nb,nc):
    """
    Return a matrix with the indices to which each atom in the
    supercell is mapped by each symmetry operation.
    """
    ntot=sposcar["positions"].shape[1]
    nat=ntot//(na*nb*nc)
    references=numpy.array(sposcar["positions"][:,:nat])
    nops=symops.translations.shape[0]
    nruter=numpy.empty((nops,ntot),dtype=numpy.int32)
    im=numpy.empty((3,nat))
    zim=numpy.empty((3,nat),dtype=numpy.int32)
    ds=numpy.empty(nat)
    for i in range(ntot):
        images=symmetry_map(sposcar["positions"][:,i],symops,na,nb,nc)
        images-=numpy.floor(images)
        for j in range(nops):
            for k in range(nat):
                im[:,k]=(images[:,j]-references[:,k])*[na,nb,nc]
                zim[:,k]=numpy.round(im[:,k])
                ds[k]=numpy.abs(im[:,k]-zim[:,k]).max()
            match=ds.argmin()
            nruter[j,i]=ind2id(zim[0,match],zim[1,match],zim[2,match],
                               match,na,nb,nc,nat)
            if ds.min()>1e-6:
                sys.exit("Error: inconsistency found when studying symmetries")
    return nruter


def wedge(poscar,symops,na,nb,nc,frange2=None):

    """
    Find out triplets with nonzero anharmonic IFCs in an irreducible wedge.
    """
    nat=poscar["positions"].shape[1]
    ntot=nat*na*nb*nc
    nops=symops.translations.shape[0]
    sposcar=gen_SPOSCAR(poscar,na,nb,nc)
    equivalences=gen_equivalences(sposcar,symops,na,nb,nc)
    if frange2==None:
        frange2=calc_frange2(sposcar)
        print "Automatic force cutoff",numpy.sqrt(frange2),"nm"
    tensor=numpy.dot(sposcar["lattvec"].T,sposcar["lattvec"])
    calc_norm2=lambda x:numpy.dot(x,numpy.dot(tensor,x))
    calc_dist2=lambda x,y:calc_norm2(x-y)
    pairs=numpy.empty((ntot,ntot),dtype=numpy.bool)
    for i in range(ntot-1):
        pairs[i,i]=True
        posi=sposcar["positions"][:,i]
        for j in range(i+1,ntot):
            d2min=numpy.inf
            for (ja,jb,jc) in itertools.product(range(-1,2),
                                                range(-1,2),
                                                range(-1,2)):
                posj=sposcar["positions"][:,j]+[ja,jb,jc]
                d2=calc_dist2(posi,posj)
                if d2<d2min:
                    d2min=d2
            pairs[i,j]=pairs[j,i]=(d2min<frange2)
    orth=numpy.empty(symops.rotations.shape)
    for i in range(nops):
        orth[i,:,:]=numpy.dot(scipy.linalg.solve(poscar["lattvec"].T,
                                                 symops.rotations[i,:,:].T),
                          poscar["lattvec"].T).T
    BB=numpy.empty((27,27))
    coeffi=numpy.empty((6*nops*27,27))
    thelist=[]
    alllist=[]
    allequilist=[]
    transformation=[]
    nindependent=[]
    independent=[]
    transformationaux=[]
    for triplet in itertools.ifilter(
            lambda t:pairs[t[0],t[1]] and pairs[t[0],t[2]] and pairs[t[1],t[2]],
            itertools.product(range(nat),range(ntot),range(ntot))):
        ii,jj,kk=triplet
        if triplet in alllist:
            continue
        thelist.append(copy.copy(triplet))
        allequilist.append([])
        transformation.append([])
        coeffi[:,:]=0.
        nnonzero=0
        for (ipermutation,perm) in enumerate(
                itertools.permutations(range(3))):
            pt=[triplet[i] for i in perm]
            for isym in range(nops):
                triplet_sym=[equivalences[isym,pt[0]],
                             equivalences[isym,pt[1]],
                             equivalences[isym,pt[2]]]
                ind0=id2ind(pt[0],na,nb,nc,nat)
                vec0=numpy.array([ind0[i] for i in ("i","j","k")])
                ind1=id2ind(pt[1],na,nb,nc,nat)
                vec1=numpy.array([ind1[i] for i in ("i","j","k")])
                ind2=id2ind(pt[2],na,nb,nc,nat)
                vec2=numpy.array([ind2[i] for i in ("i","j","k")])
                if numpy.any(vec0!=0):
                    triplet_sym[0]=ind2id(0,0,0,ind0["iat"],
                                          na,nb,nc,nat)
                    triplet_sym[1]=ind2id((vec1[0]-vec0[0])%na,
                                          (vec1[1]-vec0[1])%nb,
                                          (vec1[2]-vec0[2])%nc,
                                          ind1["iat"],
                                          na,nb,nc,nat)
                    triplet_sym[1]=ind2id((vec2[0]-vec0[0])%na,
                                          (vec2[1]-vec0[1])%nb,
                                          (vec2[2]-vec0[2])%nc,
                                          ind2["iat"],
                                          na,nb,nc,nat)
                triplet_sym=tuple(triplet_sym)
                for (ibasisprime,
                     jbasisprime,
                     kbasisprime) in itertools.product(range(3),
                                                       range(3),
                                                       range(3)):
                    indexijkprime=(3*(3*ibasisprime+jbasisprime)+
                                   kbasisprime)
                    indexrow=(27*(nops*ipermutation+isym)+
                              indexijkprime)
                    for (ibasis,
                         jbasis,
                         kbasis) in itertools.product(range(3),
                                                       range(3),
                                                       range(3)):
                        indexijk=3*(3*ibasis+jbasis)+kbasis
                        (ibasispermut,jbasispermut,kbasispermut)=[
                            (ibasis,jbasis,kbasis)[i] for i in perm]
                        BB[indexijkprime,indexijk]=(
                         orth[isym,ibasisprime,ibasispermut]*
                         orth[isym,jbasisprime,jbasispermut]*
                         orth[isym,kbasisprime,kbasispermut])
                if triplet_sym not in allequilist[-1] and (
                        ipermutation==isym==0 or triplet_sym!=triplet):
                    allequilist[-1].append(copy.copy(triplet_sym))
                    alllist.append(copy.copy(triplet_sym))
                    transformation[-1].append(BB)
                if triplet_sym==triplet:
                    for indexijk in range(27):

                        BB[indexijk,indexijk]-=1.
                        if numpy.all(BB[:,indexijk]<=1e-10):
                            continue
                        coeffi[nnonzero,:]=BB[indexijk,:]
                        nnonzero+=1
        coeffi_reduced=numpy.zeros((max(nnonzero,27),27))
        coeffi_reduced[:nnonzero,:]=coeffi[:nnonzero,:]
        rank=numpy.linalg.matrix_rank(coeffi_reduced)
        nindependent.append(rank)
        if rank==0:
            independent.append([])
            transformationaux.append(numpy.zeros((27,27)))
        else:
            Q,R,P=scipy.linalg.qr(coeffi_reduced,pivoting=True)
            iindep=P[:rank]
            independent.append(iindep)
            rematrix=numpy.zeros_like(coeffi_reduced)
            rematrix[:,iindep]=coeffi_reduced[:,iindep]
            b=scipy.linalg.lstsq(rematrix,coeffi_reduced)[0].T
            transformationaux.append(b)
    transformationaux=numpy.array(transformationaux)
    transformationarray=numpy.zeros((27,27,nops*6,len(thelist)))
    for ii in range(len(thelist)):
        for jj in range(len(allequilist[ii])):
            transformationarray[:,:nindependent[ii],jj,ii]=numpy.dot(
                transformation[ii][jj],
                transformationaux[ii][:,:nindependent[ii]])
    return (thelist,allequilist,transformationarray,independent)


def build_list4(wedgeres):
    """
    Build a list of 4-uples from the result of wedge.
    """
    ntotalindependent=sum([len(i) for i in wedgeres[3]])
    list6=[]
    for ii in range(len(wedgeres[0])):
        for jj in range(len(wedgeres[3][ii])):
            ll=wedgeres[3][ii][jj]//9
            mm=(wedgeres[3][ii][jj]%9)//3
            nn=wedgeres[3][ii][jj]%3
            list6.append(
                (ll,wedgeres[0][ii][0],
                 mm,wedgeres[0][ii][1],
                 nn,wedgeres[0][ii][2]))
    aux=collections.OrderedDict()
    for i in list6:
        fournumbers=(i[1],i[3],i[0],i[2])
        aux[fournumbers]=None
    return aux.keys()
