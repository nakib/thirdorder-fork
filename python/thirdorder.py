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
import os.path
import copy
import glob
import itertools
import contextlib
import collections
import xml.etree.cElementTree as ElementTree
import numpy
import scipy
import scipy.linalg
import thirdorder_core


H=2.116709e-3  # Magnitude of the finite displacements, in nm.


sowblock="""
.d88888b   .88888.  dP   dP   dP
88.    "' d8'   `8b 88   88   88
`Y88888b. 88     88 88  .8P  .8P
      `8b 88     88 88  d8'  d8'
d8'   .8P Y8.   .8P 88.d8P8.d8P
 Y88888P   `8888P'  8888' Y88'
ooooooooooooooooooooooooooooooooo
"""
reapblock="""
 888888ba   88888888b  .d888888   888888ba
 88    `8b  88        d8'    88   88    `8b
a88aaaa8P' a88aaaa    88aaaaa88a a88aaaa8P'
 88   `8b.  88        88     88   88
 88     88  88        88     88   88
 dP     dP  88888888P 88     88   dP
oooooooooooooooooooooooooooooooooooooooooooo
"""
doneblock="""
888888ba   .88888.  888888ba   88888888b
88    `8b d8'   `8b 88    `8b  88
88     88 88     88 88     88 a88aaaa
88     88 88     88 88     88  88
88    .8P Y8.   .8P 88     88  88
8888888P   `8888P'  dP     dP  88888888P
ooooooooooooooooooooooooooooooooooooooooo
"""


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
    nruter["na"]=na
    nruter["nb"]=nb
    nruter["nc"]=nc
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


def calc_frange(poscar,n):
    """
    Return the squared maximum distance between n-th neighbors in
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
    tonth=[]
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
            tonth.append(.5*(u[n]+u[n+1]))
        except IndexError:
            sys.stderr.write(
                "Warning: supercell too small to find n-th neighbours\n")
            tonth.append(1.1*max(u))
    return numpy.sqrt(max(tonth))


def move_two_atoms(poscar,iat,icoord,ih,jat,jcoord,jh):
    """
    Return a copy of poscar with atom iat displaced by ih nm along
    its icoord-th Cartesian coordinate and atom jat displaced by
    jh nm along its jcoord-th Cartesian coordinate.
    """
    nruter=copy.deepcopy(poscar)
    pos=numpy.dot(nruter["lattvec"],nruter["positions"][:,(iat,jat)])
    pos[icoord,0]+=ih
    pos[jcoord,1]+=jh
    nruter["positions"][:,(iat,jat)]=scipy.linalg.solve(
        nruter["lattvec"],pos)
    return nruter


def write_POSCAR(poscar,filename):
    """
    Write the contents of poscar to filename.
    """
    f=open(filename,"w")
    f.write("{}\n1.0\n".format(filename))
    for i in range(3):
        f.write("{0[0]:>20g} {0[1]:>20g} {0[2]:>20g}\n".format(
            (poscar["lattvec"][:,i]*10.).tolist()))
    f.write("{}\n".format(" ".join(poscar["elements"])))
    f.write("{}\n".format(" ".join([str(i) for i in poscar["numbers"]])))
    f.write("Direct\n")
    for i in range(poscar["positions"].shape[1]):
        f.write("{0[0]:>20g} {0[1]:>20g} {0[2]:>20g}\n".format(
            poscar["positions"][:,i].tolist()))
    f.close()


def normalize_SPOSCAR(sposcar):
    """
    Rearrange sposcar, as generated by gen_SPOSCAR, so that it is in
    valid VASP order, and return the result. Note that only positions
    are reordered, so this object should not be used in a general
    context.
    """
    nruter=copy.deepcopy(sposcar)
    # Order used internally (from most to least significant):
    # k,j,i,iat For VASP, iat must be the most significant index,
    # i.e., atoms of the same element must go together.
    indices=numpy.array(range(nruter["positions"].shape[1])).reshape(
        (sposcar["nc"],sposcar["nb"],sposcar["na"],-1))
    indices=numpy.rollaxis(indices,3,0).flatten().tolist()
    nruter["positions"]=nruter["positions"][:,indices]
    return nruter


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


def build_list4(wedgeres):
    """
    Build a list of 4-uples from the result of wedge.
    """
    ntotalindependent=sum(wedgeres["NIndependentBasis"])
    list6=[]
    for ii in range(wedgeres["Nlist"]):
        for jj in range(wedgeres["NIndependentBasis"][ii]):
            ll=wedgeres["IndependentBasis"][jj,ii]//9
            mm=(wedgeres["IndependentBasis"][jj,ii]%9)//3
            nn=wedgeres["IndependentBasis"][jj,ii]%3
            list6.append(
                (ll,wedgeres["List"][0,ii],
                 mm,wedgeres["List"][1,ii],
                 nn,wedgeres["List"][2,ii]))
    aux=collections.OrderedDict()
    for i in list6:
        fournumbers=(i[1],i[3],i[0],i[2])
        aux[fournumbers]=None
    return aux.keys()


def read_forces(filename):
    """
    Read a set of forces on atoms from filename, presumably in
    vasprun.xml format.
    """
    calculation=ElementTree.parse(filename
                                  ).getroot().find("calculation")
    for a in calculation.findall("varray"):
        if a.attrib["name"]=="forces":
            break
    nruter=[]
    for i in a.getchildren():
        nruter.append([float(j) for j in i.text.split()])
    nruter=numpy.array(nruter)
    return nruter


def build_unpermutation(sposcar):
    """
    Return a list of integers mapping the atoms in the normalized
    version of sposcar to their original indices.
    """
    indices=numpy.array(range(sposcar["positions"].shape[1])).reshape(
        (sposcar["nc"],sposcar["nb"],sposcar["na"],-1))
    indices=numpy.rollaxis(indices,3,0).flatten()
    return indices.argsort().tolist()


if __name__=="__main__":
    # TODO: allow positive and negative arguments.
    if len(sys.argv)!=6 or sys.argv[1] not in ("sow","reap"):
        sys.exit("Usage: {} sow|reap na nb nc cutoff[nm/-integer]".format(sys.argv[0]))
    action=sys.argv[1]
    na,nb,nc=[int(i) for i in sys.argv[2:5]]
    if min(na,nb,nc)<1:
        sys.exit("Error: na, nb and nc must be positive integers")
    if sys.argv[5][0]=="-":
        try:
            nneigh=-int(sys.argv[5])
        except ValueError:
            sys.exit("Error: invalid cutoff")
        if nneigh==0:
            sys.exit("Error: invalid cutoff")
    else:
        nneigh=None
        try:
            frange=float(sys.argv[5])
        except ValueError:
            sys.exit("Error: invalid cutoff")
        if frange==0.:
            sys.exit("Error: invalid cutoff")
    print "Reading POSCAR"
    poscar=read_POSCAR(".")
    natoms=len(poscar["types"])
    print "Analyzing symmetries"
    symops=thirdorder_core.SymmetryOperations(
        poscar["lattvec"],poscar["types"],
        poscar["positions"].T)
    print "- Symmetry group {} detected".format(symops.symbol)
    print "- {} symmetry operations".format(symops.translations.shape[0])
    print "Creating the supercell"
    sposcar=gen_SPOSCAR(poscar,na,nb,nc)
    ntot=natoms*na*nb*nc
    if nneigh!=None:
        frange=calc_frange(sposcar,nneigh)
        print "- Automatic cutoff: {} nm".format(frange)
    else:
        print "- User-defined cutoff: {} nm".format(frange)
    print "Calling wedge()"
    wedgeres=thirdorder_core.pywedge(poscar,sposcar,symops,frange)
    print "- {} triplet equivalence classes found".format(wedgeres["Nlist"])
    list4=build_list4(wedgeres)
    nirred=len(list4)
    nruns=4*nirred
    print "- {} DFT runs are needed".format(nruns)
    if action=="sow":
        print sowblock
        print "Writing undisplaced coordinates to 3RD.SPOSCAR"
        write_POSCAR(normalize_SPOSCAR(sposcar),"3RD.SPOSCAR")
        width=len(str(4*(len(list4)+1)))
        namepattern="3RD.POSCAR.{{0:0{}d}}".format(width)
        print "Writing displaced coordinates to 3RD.POSCAR.*"
        for i,e in enumerate(list4):
            for n in range(4):
                isign=(-1)**(n%2)
                jsign=(-1)**(n//2)
                # Start numbering the files at 1 for aesthetic
                # reasons.
                number=4*i+n+1
                dsposcar=normalize_SPOSCAR(
                    move_two_atoms(sposcar,
                                   e[0],e[2],isign*H,
                                   e[1],e[3],jsign*H))
                filename=namepattern.format(number)
                write_POSCAR(dsposcar,filename)
    else:
        print reapblock
        print "Waiting for a list of vasprun.xml files on stdin"
        filelist=[]
        for l in sys.stdin:
            s=l.strip()
            if len(s)==0:
                continue
            filelist.append(s)
        nfiles=len(filelist)
        print "- {} filenames read".format(nfiles)
        if nfiles!=nruns:
            sys.exit("Error: {} filenames were expected".
                     format(nruns))
        for i in filelist:
            if not os.path.isfile(i):
                sys.exit("Error: {} is not a regular file".
                         format(i))
        print "Reading the forces"
        p=build_unpermutation(sposcar)
        forces=[]
        for i in filelist:
            forces.append(read_forces(i)[p,:])
            print "- {} read successfully".format(i)
            res=forces[-1].sum(axis=0)
            print "- \t average residual force:"
            print "- \t {} eV/(A * atom)".format(res)
        print "Computing an irreducible set of anharmonic force constants"
        phi_part=numpy.zeros((3,nirred,ntot))
        for i,e in enumerate(list4):
            for n in range(4):
                isign=(-1)**(n%2)
                jsign=(-1)**(n//2)
                number=4*i+n
                phi_part[:,i,:]-=isign*jsign*forces[number].T
        phi_part/=(4.*H*H)
        # TODO: continue from here
    print doneblock
