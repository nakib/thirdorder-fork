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

import os
import contextlib
import numpy
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
        factor=float(f.next().strip())
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


def wedge():
    """
    Find out triplets with nonzero anharmonic IFCs in an irreducible wedge.
    """
    pass
