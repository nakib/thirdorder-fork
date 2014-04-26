#!/usr/bin/env python
# -*- coding: utf-8 -*-
#  thirdorder, help compute anharmonic IFCs from minimal sets of displacements
#  Copyright (C) 2012-2014 Wu Li <wu.li.phys2011@gmail.com>
#  Copyright (C) 2012-2014 Jesús Carrete Montaña <jcarrete@gmail.com>
#  Copyright (C) 2012-2014 Natalio Mingo Bisquert <natalio.mingo@cea.fr>
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
try:
    import cStringIO as StringIO
except ImportError:
    import StringIO
try:
    import hashlib
    hashes=True
except ImportError:
    hashes=False
import numpy
import scipy
import scipy.linalg
import scipy.spatial
import scipy.spatial.distance
import thirdorder_core


H=2.116709e-3  # Magnitude of the finite displacements, in nm.
SYMPREC=1e-5 # Tolerance for symmetry search

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
        for i in xrange(3):
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
        for i in xrange(natoms):
            nruter["positions"][:,i]=[float(j) for j in f.next().split()]
        f.close()
    nruter["types"]=[]
    for i in xrange(len(nruter["numbers"])):
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
    for pos,(k,j,i,iat) in enumerate(itertools.product(xrange(nc),
                                                       xrange(nb),
                                                       xrange(na),
                                                       xrange(
                poscar["positions"].shape[1]))):
        nruter["positions"][:,pos]=(poscar["positions"][:,iat]+[i,j,k])/[
            na,nb,nc]
    nruter["types"]=[]
    for i in xrange(na*nb*nc):
        nruter["types"].extend(poscar["types"])
    return nruter


def calc_dists(sposcar):
    """
    Return the distances between atoms in the supercells, their
    degeneracies and the associated supercell vectors.
    """
    ntot=sposcar["positions"].shape[1]
    posi=numpy.dot(sposcar["lattvec"],sposcar["positions"])
    d2s=numpy.empty((27,ntot,ntot))
    for j,(ja,jb,jc) in enumerate(itertools.product(xrange(-1,2),
                                                    xrange(-1,2),
                                                    xrange(-1,2))):
        posj=numpy.dot(sposcar["lattvec"],(sposcar["positions"].T+[ja,jb,jc]).T)
        d2s[j,:,:]=scipy.spatial.distance.cdist(posi.T,posj.T,"sqeuclidean")
    d2min=d2s.min(axis=0)
    dmin=numpy.sqrt(d2min)
    degenerate=(numpy.abs(d2s-d2min)<1e-4)
    nequi=degenerate.sum(axis=0)
    maxequi=nequi.max()
    shifts=numpy.empty((ntot,ntot,maxequi))
    sorting=numpy.argsort(numpy.logical_not(degenerate),axis=0)
    shifts=numpy.transpose(sorting[:maxequi,:,:],(1,2,0))
    return (dmin,nequi,shifts)


def calc_frange(poscar,sposcar,n,dmin):
    """
    Return the maximum distance between n-th neighbors in the structure.
    """
    natoms=len(poscar["types"])
    tonth=[]
    warned=False
    for i in xrange(natoms):
        ds=dmin[i,:].tolist()
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
    return max(tonth)


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
    global hashes
    f=StringIO.StringIO()
    f.write("1.0\n".format(filename))
    for i in xrange(3):
        f.write("{0[0]:>20.15f} {0[1]:>20.15f} {0[2]:>20.15f}\n".format(
            (poscar["lattvec"][:,i]*10.).tolist()))
    f.write("{}\n".format(" ".join(poscar["elements"])))
    f.write("{}\n".format(" ".join([str(i) for i in poscar["numbers"]])))
    f.write("Direct\n")
    for i in xrange(poscar["positions"].shape[1]):
        f.write("{0[0]:>20.15f} {0[1]:>20.15f} {0[2]:>20.15f}\n".format(
            poscar["positions"][:,i].tolist()))
    if hashes:
        header=hashlib.sha1(f.getvalue()).hexdigest()
    else:
        header=filename
    with open(filename,"w") as finalf:
        finalf.write("{}\n".format(header))
        finalf.write(f.getvalue())
    f.close()


def normalize_SPOSCAR(sposcar):
    """
    Rearrange sposcar, as generated by gen_SPOSCAR, so that it is in
    valid VASP order, and return the result.
    """
    nruter=copy.deepcopy(sposcar)
    # Order used internally (from most to least significant):
    # k,j,i,iat For VASP, iat must be the most significant index,
    # i.e., atoms of the same element must go together.
    indices=numpy.array(xrange(nruter["positions"].shape[1])).reshape(
        (sposcar["nc"],sposcar["nb"],sposcar["na"],-1))
    indices=numpy.rollaxis(indices,3,0).flatten().tolist()
    nruter["positions"]=nruter["positions"][:,indices]
    nruter["types"].sort()
    return nruter


def build_list4(wedgeres):
    """
    Build a list of 4-uples from the result of wedge.
    """
    ntotalindependent=sum(wedgeres["NIndependentBasis"])
    list6=[]
    for ii in xrange(wedgeres["Nlist"]):
        for jj in xrange(wedgeres["NIndependentBasis"][ii]):
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
    indices=numpy.array(xrange(sposcar["positions"].shape[1])).reshape(
        (sposcar["nc"],sposcar["nb"],sposcar["na"],-1))
    indices=numpy.rollaxis(indices,3,0).flatten()
    return indices.argsort().tolist()


def write_ifcs(phifull,poscar,sposcar,dmin,nequi,shifts,frange,filename):
    """
    Write out the full anharmonic interatomic force constant matrix,
    taking the force cutoff into account.
    """
    natoms=len(poscar["types"])
    ntot=len(sposcar["types"])

    shifts27=list(itertools.product(xrange(-1,2),
                                    xrange(-1,2),
                                    xrange(-1,2)))
    frange2=frange*frange

    nblocks=0
    f=StringIO.StringIO()
    for ii,jj in itertools.product(xrange(natoms),
                                   xrange(ntot)):
        if dmin[ii,jj]>=frange:
            continue
        jatom=jj%natoms
        shiftsij=[shifts27[i] for i in shifts[ii,jj,:nequi[ii,jj]]]
        for kk in xrange(ntot):
            if dmin[ii,kk]>=frange:
                continue
            katom=kk%natoms
            shiftsik=[shifts27[i] for i in shifts[ii,kk,:nequi[ii,kk]]]
            d2min=numpy.inf
            for shift2 in shiftsij:
                carj=numpy.dot(sposcar["lattvec"],shift2+sposcar["positions"][:,jj])
                for shift3 in shiftsik:
                    cark=numpy.dot(sposcar["lattvec"],shift3+sposcar["positions"][:,kk])
                    d2=((carj-cark)**2).sum()
                    if d2<d2min:
                        best2=shift2
                        best3=shift3
                        d2min=d2
            if d2min>=frange2:
                continue
            nblocks+=1
            Rj=numpy.dot(sposcar["lattvec"],
                         best2+sposcar["positions"][:,jj]-sposcar["positions"][:,jatom])
            Rk=numpy.dot(sposcar["lattvec"],
                         best3+sposcar["positions"][:,kk]-sposcar["positions"][:,katom])
            f.write("\n")
            f.write("{:>5}\n".format(nblocks))
            f.write("{0[0]:>15.10e} {0[1]:>15.10e} {0[2]:>15.10e}\n".
                    format(list(10.*Rj)))
            f.write("{0[0]:>15.10e} {0[1]:>15.10e} {0[2]:>15.10e}\n".
                    format(list(10.*Rk)))
            f.write("{:>6d} {:>6d} {:>6d}\n".format(ii+1,jatom+1,katom+1))
            for ll,mm,nn in itertools.product(xrange(3),
                                              xrange(3),
                                              xrange(3)):
                f.write("{:>2d} {:>2d} {:>2d} {:>20.10e}\n".
                        format(ll+1,mm+1,nn+1,phifull[ll,mm,nn,ii,jj,kk]))
    ffinal=open(filename,"w")
    ffinal.write("{:>5}\n".format(nblocks))
    ffinal.write(f.getvalue())
    f.close()
    ffinal.close()


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
    print "Analyzing the symmetries"
    symops=thirdorder_core.SymmetryOperations(
        poscar["lattvec"],poscar["types"],
        poscar["positions"].T,SYMPREC)
    print "- Symmetry group {} detected".format(symops.symbol)
    print "- {} symmetry operations".format(symops.translations.shape[0])
    print "Creating the supercell"
    sposcar=gen_SPOSCAR(poscar,na,nb,nc)
    ntot=natoms*na*nb*nc
    print "Computing all distances in the supercell"
    dmin,nequi,shifts=calc_dists(sposcar)
    if nneigh!=None:
        frange=calc_frange(poscar,sposcar,nneigh,dmin)
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
            for n in xrange(4):
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
            for n in xrange(4):
                isign=(-1)**(n//2)
                jsign=-(-1)**(n%2)
                number=nirred*n+i
                phipart[:,i,:]-=isign*jsign*forces[number].T
        phipart/=(400.*H*H)
        print "Reconstructing the full matrix"
        phifull=thirdorder_core.reconstruct_ifcs(phipart,wedgeres,list4,poscar,sposcar)
        print "Writing the constants to FORCE_CONSTANTS_3RD"
        write_ifcs(phifull,poscar,sposcar,dmin,nequi,shifts,frange,"FORCE_CONSTANTS_3RD")
    print doneblock
