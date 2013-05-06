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
import collections
import numpy
import spg
import common

if __name__=="__main__":
    if len(sys.argv)<2 or sys.argv[1] not in ("sow","reap"):
        sys.exit("Usage: {} sow|reap [options]".format(sys.argv[0]))
    action=sys.argv[1]
    if action=="sow":
        if len(sys.argv)!=5:
            sys.exit("Usage: {} sow na nb nc".format(sys.argv[0]))
        na,nb,nc=[int(i) for i in sys.argv[2:]]
        poscar=common.read_POSCAR(".")
        symops=spg.SymmetryOperations(poscar["lattvec"],poscar["types"],
                                      poscar["positions"].T)
        print "Symmetry group {} detected".format(symops.symbol)
        wedgeres=common.wedge(poscar,symops,na,nb,nc)
        ntotalindependent=sum([len(i) for i in wedgeres[3]])

        list6=[]
        for ii in range(len(wedgeres[0])):
            for jj in range(len(wedgeres[3][ii])):
                ll=(wedgeres[3][ii][jj]-1)//9
                mm=((wedgeres[3][ii][jj]-1)//9)%3
                nn=(wedgeres[3][ii][jj]-1)%3
                list6.append(
                    (ll,wedgeres[0][ii][0],
                     mm,wedgeres[0][ii][1],
                     nn,wedgeres[0][ii][2]))
        aux=collections.OrderedDict()
        for x in list6:
            aux[x[:4]]=True
        list4=aux.keys()
        print len(list4)
        f=open("List_4.d","w")
        for i in list4:
            f.write("{0[1]} {0[3]} {0[0]} {0[2]}\n".format(i))
        f.close()
        f=open("List_6.d","w")
        for i in list6:
            f.write("{0[0]} {0[1]} {0[2]} {0[3]} {0[4]} {0[5]}\n".format(i))
        f.close()
    else:
        pass
