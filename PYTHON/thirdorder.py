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
import spg
import common

if __name__=="__main__":
    if len(sys.argv)<2 or sys.argv[1] not in ("seed","harvest"):
        sys.exit("Usage: {} seed|harvest [options]".format(sys.argv[0]))
    action=sys.argv[1]
    if action=="seed":
        if len(sys.argv)!=5:
            sys.exit("Usage: {} seed na nb nc".format(sys.argv[0]))
        na,nb,nc=[float(i) for i in sys.argv[2:]]
        poscar=common.read_POSCAR(".")
        symops=spg.SymmetryOperations(poscar["lattvec"].T,poscar["types"],
                                      poscar["positions"].T)
        print "Symmetry group {} detected".format(symops.symbol)
    else:
        pass
