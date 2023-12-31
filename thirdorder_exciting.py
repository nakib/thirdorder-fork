#!/usr/bin/env python
# -*- coding: utf-8 -*-
#  thirdorder, help compute anharmonic IFCs from minimal sets of displacements
#  Copyright (C) 2012-2018 Wu Li <wu.li.phys2011@gmail.com>
#  Copyright (C) 2012-2018 Jesús Carrete Montaña <jcarrete@gmail.com>
#  Copyright (C) 2012-2018 Natalio Mingo Bisquert <natalio.mingo@cea.fr>
#  Copyright (C) 2014-2018 Antti J. Karttunen <antti.j.karttunen@iki.fi>
#  Copyright (C) 2016-2018 Genadi Naydenov <gan503@york.ac.uk>
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

from __future__ import print_function
try:
    xrange
except NameError:
    xrange = range

import sys
import os.path
import glob
try:
    from lxml import etree as ElementTree
    xmllib = "lxml.etree"
except ImportError:
    try:
        import xml.etree.cElementTree as ElementTree
        xmllib = "cElementTree"
    except ImportError:
        import xml.etree.ElementTree as ElementTree
        xmllib = "ElementTree"
try:
    import cStringIO as StringIO
except ImportError:
    try:
        import StringIO
    except ImportError:
        import io as StringIO
try:
    import hashlib
    hashes = True
except ImportError:
    hashes = False

import thirdorder_core
from thirdorder_common import *


def read_exciting_input(directory):
    """
    Reads and returns all the relevant information from exciting input.xml file.
    """

    print('Parsing exciting input.xml...')
    
    unitcell_info = dict()
    
    with dir_context(directory):
        unitcell_info['lattvec'] = np.empty((3, 3))
        
        tree = ElementTree.parse('input.xml')
        root = tree.getroot()

        for structure in root.iter('structure'):
            lattvecs = []
            for crystal in structure.findall('crystal'):
                #The 0.1 is for Ang -> nm conversion
                scale = float(crystal.attrib['scale'])*0.1
                for basevect in crystal.findall('basevect'):
                    lattvecs.append([scale*float(a) for a in basevect.text.split()])

                for dim in range(3):
                    unitcell_info['lattvec'][:, dim] = lattvecs[dim]

            elements = []
            unique_elements = []
            basis = []
            types = []
            numatoms = 0
            numspecies = 0
            for species in structure.findall('species'):
                element = species.attrib['speciesfile'].split(".")[0]
                unique_elements.append(element)
                numspecies += 1
                for atom in species.findall('atom'):
                    numatoms += 1
                    basis.append([float(a) for a in atom.attrib['coord'].split()])
                    elements.append(element)
                    types.append(numspecies)

            unitcell_info['elements'] = elements
            unitcell_info['unique_elements'] = unique_elements
            unitcell_info['positions'] = np.empty((3, numatoms))
            for a in range(numatoms):
                unitcell_info['positions'][:, a] = basis[a]
            print(unitcell_info['positions'])
            unitcell_info['types'] = types

        print('Lattice vectors: ' + str(unitcell_info['lattvec']) + ' nm')
        #print('Elements: ' + str(unitcell_info['elements']))
        print('Unique elements: ' + str(unitcell_info['unique_elements']))
        print('Number of atoms = ' + str(numatoms))
        print('Basis vectors = ' + str(unitcell_info['positions'].T))
        print('Atom types = ' + str(unitcell_info['types']))

    return unitcell_info


def gen_supercell(unitcell_info, na, nb, nc):
    """
    Create a dictionary similar to the first argument but describing a
    supercell.
    """
    nruter = dict()
    nruter["na"] = na
    nruter["nb"] = nb
    nruter["nc"] = nc
    nruter["lattvec"] = np.array(unitcell_info["lattvec"])
    nruter["lattvec"][:, 0] *= na
    nruter["lattvec"][:, 1] *= nb
    nruter["lattvec"][:, 2] *= nc
    nruter["elements"] = []
    nruter["types"] = []
    nruter["positions"] = np.empty(
        (3, unitcell_info["positions"].shape[1] * na * nb * nc))
    pos = 0
    for pos, (k, j, i, iat) in enumerate(
            itertools.product(xrange(nc), xrange(nb), xrange(na),
                              xrange(unitcell_info["positions"].shape[1]))):
        nruter["positions"][:, pos] = (unitcell_info["positions"][:, iat] +
                                       [i, j, k]) / [na, nb, nc]
        nruter["elements"].append(unitcell_info["elements"][iat])
        nruter["types"].append(unitcell_info["types"][iat])
    return nruter


def write_supercell(templatefile, supercell, filename, number):
    """
    Create an exciting input file for a supercell calculation
    from a template.
    """

    #Number of atoms in supercell
    numatoms_sc = len(supercell['positions'][0, :])
    
    tree = ElementTree.parse(templatefile)
    root = tree.getroot()

    for structure in root.iter('structure'):
        for crystal in structure.findall('crystal'):
            #Write new scale
            crystal.attrib['scale'] = "0.10000"
            #Remove existing unitcell lattice vectors
            for basevect in crystal.findall('basevect'):
                crystal.remove(basevect)
            #Add supercell basis bectors
            for i in range(3):
                ElementTree.SubElement(crystal, 'basevect').text = \
                    '    '.join(map(str, supercell["lattvec"][:, i]))
            
            numspecies = 0
            for species in structure.findall('species'):
                element = species.attrib['speciesfile'].split(".")[0]
                numspecies += 1

                #Remove existing unitcell basis
                for atom in species.findall('atom'):
                    species.remove(atom)

                #Write supercell basis
                for i in range(numatoms_sc):
                    if(supercell['types'][i] == numspecies):
                        atom = ElementTree.SubElement(species, 'atom')
                        atom.set('coord', '    '.join(map(str, supercell["positions"][:, i])))

    #Need python v>3.9 for this to work? The output without this will be tots uggo!
    #ElementTree.indent(tree, '    ')

    #tree.write(filename, pretty_print = True, encoding='UTF-8')
    tree.write(filename)


#TODO
#def read_forces(filename):
#    """
#    Read a set of forces on atoms from filename, in
#    exciting's output format. Units: eV/nm
#    """
    

if __name__ == '__main__':
    def usage():
        """
        Print an usage message and exit.
        """
        sys.exit("""Usage:
\t{program:s} unitcell.in sow na nb nc cutoff[nm/-integer] supercell_template.in
\t{program:s} unitcell.in reap na nb nc cutoff[nm/-integer]""".format(
            program=sys.argv[0]))

    if len(sys.argv) not in (7, 8) or sys.argv[2] not in ("sow", "reap"):
        usage()
    ufilename = sys.argv[1]
    action = sys.argv[2]
    na, nb, nc = [int(i) for i in sys.argv[3:6]]
    if action == "sow":
        if len(sys.argv) != 8:
            usage()
        sfilename = sys.argv[7]
    else:
        if len(sys.argv) != 7:
            usage()
    if min(na, nb, nc) < 1:
        sys.exit("Error: na, nb and nc must be positive integers")
    if sys.argv[6][0] == "-":
        try:
            nneigh = -int(sys.argv[6])
        except ValueError:
            sys.exit("Error: invalid cutoff")
        if nneigh == 0:
            sys.exit("Error: invalid cutoff")
    else:
        nneigh = None
        try:
            frange = float(sys.argv[6])
        except ValueError:
            sys.exit("Error: invalid cutoff")
        if frange == 0.:
            sys.exit("Error: invalid cutoff")
    
    print("Reading {0}".format(ufilename))    

    unitcell_info = read_exciting_input('./')

    natoms = len(unitcell_info['types'])

    print('Analyzing the symmetries')
    symops = thirdorder_core.SymmetryOperations(
        unitcell_info["lattvec"], unitcell_info["types"], unitcell_info["positions"].T, SYMPREC)
    print("- Symmetry group {0} detected".format(symops.symbol))
    print("- {0} symmetry operations".format(symops.translations.shape[0]))

    print("Creating the supercell")
    supercell = gen_supercell(unitcell_info, na, nb, nc)
    ntot = natoms * na * nb * nc

    print("Computing all distances in the supercell")
    dmin, nequi, shifts = calc_dists(supercell)
    if nneigh != None:
        frange = calc_frange(unitcell_info, supercell, nneigh, dmin)
        print("- Automatic cutoff: {0} nm".format(frange))
    else:
        print("- User-defined cutoff: {0} nm".format(frange))

    print("Looking for an irreducible set of third-order IFCs")
    wedge = thirdorder_core.Wedge(unitcell_info, supercell, symops, dmin, nequi, shifts,
                                  frange)
    print("- {0} triplet equivalence classes found".format(wedge.nlist))

    list4 = wedge.build_list4()
    nirred = len(list4)
    nruns = 4 * nirred
    print("- {0} DFT runs are needed".format(nruns))

    if action == "sow":
        print(sowblock)
        print("Writing undisplaced coordinates to BASE.{0}".format(
            os.path.basename(sfilename)))
        write_supercell(sfilename, supercell,
                        "BASE.{0}".format(os.path.basename(sfilename)), 0)
        width = len(str(4 * (len(list4) + 1)))
        namepattern = "DISP.{0}.{{0:0{1}d}}".format(
            os.path.basename(sfilename), width)
        print("Writing displaced coordinates to DISP.{0}.*".format(
            os.path.basename(sfilename)))
        for i, e in enumerate(list4):
            for n in xrange(4):
                isign = (-1)**(n // 2)
                jsign = -(-1)**(n % 2)
                # Start numbering the files at 1 for aesthetic
                # reasons.
                number = nirred * n + i + 1
                dsupercell = move_two_atoms(supercell, e[1], e[3], isign * H, e[0],
                                          e[2], jsign * H)
                filename = namepattern.format(number)
                write_supercell(sfilename, dsupercell, filename, number)
