# Parser for xyz files in this very specific case. The files also have atom indices of each ligand in the complex. 
# The coordinating atoms are also listed as lc, which stands for ligand center.
# Although it might seem rather tedious, regex will be used.

import re
import numpy as np
from .elements import *
from .bonds import *

coord_pattern = re.compile("""\"[A-Za-z*]+\",\s*-*\d*.\d*,\s*-*\d*.\d*,\s*-*\d*.\d*""")
ind_pattern = re.compile("{\d*,\s*\d*,\s*{\d\d*,\s*(?:\d\d*,\s*)*\d*},\s*{\d\d*,\s*(?:\d\d*,\s*)*\d*}")

class simple_mol(object):

    def __init__(self, filename: str):
        f = open(filename, 'r')
        self.text = f.read().replace('\n', '')
    
    def get_coords(self):

        """
        Parse the coordinates from the xyz file.
        """

        coords_lines = coord_pattern.findall(self.text)
        def helper(line):
            line = line.replace('"', '')
            line = line.split(',')
            atom = line[0]
            coords = np.array([float(line[i]) for i in range(1,4)])
            return atom, coords
        atom, coords = helper(coords_lines[0])
        atoms = [atom]
        coords_all = np.array([coords])
        for i in range(1, len(coords_lines)):
            atom, coords = helper(coords_lines[i])
            atoms.append(atom)
            coords_all = np.concatenate((coords_all, np.array([coords])), axis=0)
        self.mc = []
        for i, atom in enumerate(atoms):
            if ismetal(atom):
                self.mc.append(i)
        self.atoms = atoms
        self.coords_all = coords_all
        return self
    
    def get_ligand_ind(self):

        """
        Parse the ligand indices from the xyz file.
        """

        ind_lines = ind_pattern.findall(self.text)
        def helper(line):
            line = line.translate({ord(i): None for i in "{} "}).split(',')
            ind = [int(i)-1 for i in line] # int(i)-1 is included because the indices are generated in Mathematica.
            lcs = ind[:2]
            return lcs, ind
        self.lcs = []
        self.ligand_ind = []
        for line in ind_lines:
            lcs, ind = helper(line)
            self.lcs.append(lcs)
            self.ligand_ind.append(ind)
        return self

    def parse_all(self):
        self = self.get_coords()
        self = self.get_ligand_ind()
        del self.text
        return self

    
    def get_graph(self):
        """
        Only support one-metal complexes currently.
        """
        if len(self.mc) == 1:
            self = get_graph_by_ligands(self)
        return self
    