#!/usr/bin/env python
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
import tempfile
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
        nruter["positions"]=scipy.linalg.solve(nruter["lattvec"],
                                               nruter["positions"]*factor)
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


def calc_dists2(poscar,sposcar):
    """
    Return a matrix with the squared distances between atoms
    in the supercell, using normal images.

    """
    natoms=poscar["positions"].shape[1]
    ntot=sposcar["positions"].shape[1]
    tensor=numpy.dot(sposcar["lattvec"].T,sposcar["lattvec"])
    calc_norm2=lambda x:numpy.dot(x,numpy.dot(tensor,x))
    calc_dist2=lambda x,y:calc_norm2(x-y)
    d2=numpy.empty((natoms,ntot))
    for i in range(natoms):
        posi=sposcar["positions"][:,i]
        for j in range(ntot):
            d2min=numpy.inf
            for (ja,jb,jc) in itertools.product(range(-1,2),
                                                range(-1,2),
                                                range(-1,2)):
                posj=sposcar["positions"][:,j]+[ja,jb,jc]
                d2new=calc_dist2(posi,posj)
                if d2new<d2min:
                    d2min=d2new
            d2[i,j]=d2min
    return d2


def calc_frange(poscar,sposcar,n):
    """
    Return the squared maximum distance between n-th neighbors in
    the structure.
    """
    natoms=poscar["positions"].shape[1]
    d2=calc_dists2(poscar,sposcar)
    tonth=[]
    warned=False
    for i in range(natoms):
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
            if not warned:
                sys.stderr.write(
                    "Warning: supercell too small to find n-th neighbours\n")
                warned=True
            tonth.append(1.1*max(u))
    return numpy.sqrt(max(tonth))


def move_two_atoms(poscar,iat,icoord,ih,jat,jcoord,jh):
    """
    Return a copy of poscar with atom iat displaced by ih nm along
    its icoord-th Cartesian coordinate and atom jat displaced by
    jh nm along its jcoord-th Cartesian coordinate.
    """
    nruter=copy.deepcopy(poscar)
    disp=numpy.zeros(3)
    disp[icoord]=ih
    nruter["positions"][:,iat]+=scipy.linalg.solve(nruter["lattvec"],
                                                   disp)
    disp[:]=0.
    disp[jcoord]=jh
    nruter["positions"][:,jat]+=scipy.linalg.solve(nruter["lattvec"],
                                                   disp)
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


def reconstruct_ifcs(phipart,wedgeres,list4,poscar,sposcar):
    """
    Recover the full anharmonic IFC matrix from the irreducible set of
    force constants.
    """
    natoms=len(poscar["types"])
    ntot=len(sposcar["types"])
    nruter=numpy.zeros((3,3,3,natoms,ntot,ntot))
    naccumindependent=numpy.insert(numpy.cumsum(wedgeres["NIndependentBasis"]),
                                   0,[0])
    ntotalindependent=naccumindependent[-1]
    for i,e in enumerate(list4):
        nruter[e[2],e[3],:,e[0],e[1],:]=phipart[:,i,:]
    philist=[]
    for ii in range(wedgeres["Nlist"]):
        for jj in range(wedgeres["NIndependentBasis"][ii]):
            ll=wedgeres["IndependentBasis"][jj,ii]//9
            mm=(wedgeres["IndependentBasis"][jj,ii]%9)//3
            nn=wedgeres["IndependentBasis"][jj,ii]%3
            philist.append(nruter[ll,mm,nn,
                                  wedgeres["List"][0,ii],
                                  wedgeres["List"][1,ii],
                                  wedgeres["List"][2,ii]])
    philist=numpy.array(philist)
    ind1equi=numpy.zeros((natoms,ntot,ntot))
    ind2equi=numpy.zeros((natoms,ntot,ntot))
    for ii in range(wedgeres["Nlist"]):
        for jj in range(wedgeres["Nequi"][ii]):
            ind1equi[wedgeres["ALLEquiList"][0,jj,ii],
                     wedgeres["ALLEquiList"][1,jj,ii],
                     wedgeres["ALLEquiList"][2,jj,ii]]=ii
            ind2equi[wedgeres["ALLEquiList"][0,jj,ii],
                     wedgeres["ALLEquiList"][1,jj,ii],
                     wedgeres["ALLEquiList"][2,jj,ii]]=jj
    aa=numpy.zeros((natoms*ntot*27,ntotalindependent))
    nnonzero=0
    nonzerolist=[]
    for ii,jj,ll,mm,nn in itertools.product(range(natoms),
                                            range(ntot),
                                            range(3),
                                            range(3),
                                            range(3)):
        tribasisindex=(ll*3+mm)*3+nn
        rowindex=(ii*natoms+jj)*27+tribasisindex
        for kk,ix in itertools.product(range(ntot),
                                       range(wedgeres["Nlist"])):
            if ind1equi[ii,jj,kk]==ix:
                aa[rowindex,naccumindependent[ix]:naccumindependent[ix+1]
                   ]+=wedgeres["TransformationArray"][
                       tribasisindex,:wedgeres["NIndependentBasis"][ix],
                       ind2equi[ii,jj,kk],ix]
        aa[rowindex,aa[rowindex,:]<=1e-14]=0.
        aa[nnonzero,:ntotalindependent]=aa[rowindex,:ntotalindependent]
        nnonzero+=1
    aux=numpy.array(aa[:nnonzero,:ntotalindependent])
    gaussianres=thirdorder_core.pygaussian(aux)
    aux=gaussianres["a"]
    nnonzero=gaussianres["NIndependent"]
    bb=numpy.array(aux[:nnonzero,:ntotalindependent]).T
    multiplier=-scipy.linalg.lstsq(bb,philist)[0]
    compensation=numpy.dot(bb,multiplier)
    nruter[:,:,:,:,:,:]=0.
    for ii in range(wedgeres["Nlist"]):
        for jj in range(wedgeres["Nequi"][ii]):
            for ll,mm,nn in itertools.product(range(3),
                                              range(3),
                                              range(3)):
                tribasisindex=(ll*3+mm)*3+nn
                for ix in range(wedgeres["NIndependentBasis"][ii]):
                    nruter[ll,mm,nn,wedgeres["ALLEquiList"][0,jj,ii],
                           wedgeres["ALLEquiList"][1,jj,ii],
                           wedgeres["ALLEquiList"][2,jj,ii]
                           ]+=wedgeres["TransformationArray"][
                               tribasisindex,ix,jj,ii]*philist[
                                   naccumindependent[ii]+ix]
    return nruter


def write_ifcs(phifull,poscar,sposcar,frange,filename):
    """
    Write out the full anharmonic interatomic force constant matrix,
    taking the force cutoff into account.
    """
    frange2=frange*frange
    natoms=len(poscar["types"])
    ntot=len(sposcar["types"])

    nblocks=0
    tmpname=tempfile.mkstemp()[1]
    f=open(tmpname,"w")

    tensor=numpy.dot(sposcar["lattvec"].T,sposcar["lattvec"])
    calc_norm2=lambda x:numpy.dot(x,numpy.dot(tensor,x))
    calc_dist2=lambda x,y:calc_norm2(x-y)
    nruter=[]

    shift2all=numpy.zeros((3,27),dtype=numpy.int32)
    shift3all=numpy.zeros((3,27),dtype=numpy.int32)
    d2s=numpy.zeros(27)
    for ii in range(natoms):
        posi=sposcar["positions"][:,ii]
        for jj in range(ntot):
            for i,(ja,jb,jc) in enumerate(itertools.product(range(-1,2),
                                                            range(-1,2),
                                                            range(-1,2))):
                posj=sposcar["positions"][:,jj]+[ja,jb,jc]
                d2s[i]=calc_dist2(posi,posj)
            d2min=d2s.min()
            n2equi=0
            for i,(ja,jb,jc) in enumerate(itertools.product(range(-1,2),
                                                            range(-1,2),
                                                            range(-1,2))):
                if numpy.abs(d2s[i]-d2min)<1e-2:
                    shift2all[:,n2equi]=[ja,jb,jc]
                    n2equi+=1
            if d2min>=frange2:
                continue
            for kk in range(ntot):
                n3equi=0
                for i,(ja,jb,jc) in enumerate(itertools.product(range(-1,2),
                                                                range(-1,2),
                                                                range(-1,2))):
                    posk=sposcar["positions"][:,kk]+[ja,jb,jc]
                    d2s[i]=calc_dist2(posi,posk)
                d2min=d2s.min()
                n3equi=0
                for i,(ja,jb,jc) in enumerate(itertools.product(range(-1,2),
                                                                range(-1,2),
                                                                range(-1,2))):
                    if numpy.abs(d2s[i]-d2min)<1e-2:
                        shift3all[:,n3equi]=[ja,jb,jc]
                        n3equi+=1
                if d2min>=frange2:
                    continue
                dp2min=numpy.inf
                for iaux in range(n2equi):
                    for jaux in range(n3equi):
                        dp2=calc_dist2(sposcar["positions"][:,jj]+shift2all[:,iaux],
                                       sposcar["positions"][:,kk]+shift3all[:,jaux])
                        if dp2<dp2min:
                            dp2min=dp2
                            shift2=shift2all[:,iaux]
                            shift3=shift3all[:,jaux]
                if dp2min>=frange2:
                    continue
                nblocks+=1
                jatom=jj%natoms
                katom=kk%natoms
                carj=(numpy.dot(sposcar["lattvec"],
                                shift2+sposcar["positions"][:,jj])-
                                numpy.dot(poscar["lattvec"],
                                          poscar["positions"][:,jatom]))
                cark=(numpy.dot(sposcar["lattvec"],
                                shift3+sposcar["positions"][:,kk])-
                                numpy.dot(poscar["lattvec"],
                                          poscar["positions"][:,katom]))
                f.write("\n")
                f.write("{:>5}\n".format(nblocks))
                f.write("{0[0]:>15.10e} {0[1]:>15.10e} {0[2]:>15.10e}\n".
                        format(list(10.*carj)))
                f.write("{0[0]:>15.10e} {0[1]:>15.10e} {0[2]:>15.10e}\n".
                        format(list(10.*cark)))
                f.write("{:>6d} {:>6d} {:>6d}\n".format(ii+1,jatom+1,katom+1))
                for ll,mm,nn in itertools.product(range(3),
                                                  range(3),
                                                  range(3)):
                    f.write("{:>2d} {:>2d} {:>2d} {:>20.10e}\n".
                            format(ll+1,mm+1,nn+1,phifull[ll,mm,nn,ii,jj,kk]))
    f.close()
    f=open(filename,"w")
    f.write("{:>5}\n".format(nblocks))
    ftmp=open(tmpname,"r")
    for l in ftmp:
        f.write(l)
    ftmp.close()
    f.close()


if __name__=="__main__":
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
        frange=calc_frange(poscar,sposcar,nneigh)
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
                isign=(-1)**(n//2)
                jsign=-(-1)**(n%2)
                # Start numbering the files at 1 for aesthetic
                # reasons.
                number=nirred*n+i+1
                dsposcar=normalize_SPOSCAR(
                    move_two_atoms(sposcar,
                                   e[1],e[3],isign*H,
                                   e[0],e[2],jsign*H))
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
            print "- \t Average residual force:"
            print "- \t {} eV/(A * atom)".format(res)
        print "Computing an irreducible set of anharmonic force constants"
        phipart=numpy.zeros((3,nirred,ntot))
        for i,e in enumerate(list4):
            for n in range(4):
                isign=(-1)**(n//2)
                jsign=-(-1)**(n%2)
                number=nirred*n+i
                phipart[:,i,:]-=isign*jsign*forces[number].T
        phipart/=(400.*H*H)
        print "Reconstructing the full matrix"
        phifull=reconstruct_ifcs(phipart,wedgeres,list4,poscar,sposcar)
        print "Writing the constants to FORCE_CONSTANTS_3RD"
        write_ifcs(phifull,poscar,sposcar,frange,"FORCE_CONSTANTS_3RD")
    print doneblock
