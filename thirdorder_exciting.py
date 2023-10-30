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
    
    crystal_info = dict()
        
    with dir_context(directory):
        crystal_info['lattvec'] = np.empty((3, 3))
        
        tree = ElementTree.parse('input.xml')
        root = tree.getroot()

        for structure in root.iter('structure'):
            lattvecs = []
            for crystal in structure.findall('crystal'):
                scale = float(crystal.attrib['scale'])
                for basevect in crystal.findall('basevect'):
                    lattvecs.append([scale*float(a) for a in basevect.text.split()])

                crystal_info['lattvec'] = lattvecs

            elements = []
            for species in structure.findall('species'):
                elements.append(species.attrib['speciesfile'].split(".")[0])

            crystal_info['elements'] = elements

        print('Lattice vectors: ' + str(crystal_info['lattvec']) + ' Bohr')
        print('Elements: ' + str(crystal_info['elements']))

    return crystal_info
            
if __name__ == '__main__':
    read_exciting_input('./')
