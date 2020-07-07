import re
import numpy as np
from .elements import *
from collections import defaultdict
from scipy.spatial import distance_matrix

coord_pattern = re.compile("""\"[A-Za-z*]+\",\s*-*\d*.\d*,\s*-*\d*.\d*,\s*-*\d*.\d*""")
ind_pattern = re.compile("{\d*,\s*\d*,\s*{\d\d*,\s*(?:\d\d*,\s*)*\d*},\s*{\d\d*,\s*(?:\d\d*,\s*)*\d*}")

class simple_mol(object):

    """
    A very simple molecule object. Parses information from a specific type of xyz file. Not generic.
    The coordinating atoms are also listed as lc, which stands for ligand center.
    Although it might seem rather tedious, regex will be used.
    """

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
        self.mcs = []
        if ismetal(atom):
            self.mcs.append(0)
        for i in range(1, len(coords_lines)):
            atom, coords = helper(coords_lines[i])
            atoms.append(atom)
            if ismetal(atom):
                self.mcs.append(i)
            coords_all = np.concatenate((coords_all, np.array([coords])), axis=0)
       
        self.atoms = atoms
        self.natoms = len(atoms)
        self.coords_all = coords_all
    
    def get_ligand_ind(self):

        """
        Parse the ligand indices from the xyz file. This only works for the very specific format that we have right now.
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

    def init_distances(self):
        self.distances = self.graph.copy()

    def get_all_distances(self, depth: int):

        """
        Only calculates shortest-path distances up to depth because we do not care about long range autocorrelations yet.
        """

        for atom in range(self.natoms):
            bfs_distances(self, atom, depth)

    def parse_with_ind(self):

        """
        The simplest case where indices of ligand atoms are available. Call this first before calculating RAC features.
        """

        self.get_coords()
        self.get_ligand_ind()
        del self.text
        self.graph = get_graph_by_ligands(self)
        self.coordination = self.graph.sum(axis=0)
        self.init_distances()
    
    def parse_all(self):

        """
        A more general case.
        """

        self.get_coords()
        del self.text
        self.graph = get_graph_full_scope(self)
        bfs_ligands(self)
        self.coordination = self.graph.sum(axis=0)
        self.init_distances()

    def get_bonded_atoms(self, atom_index: int):
        con = self.graph[atom_index]
        return np.where(con==1)[0]

    def get_bonded_atoms_multiple(self, atom_ind: list) -> list:
        ind = set(atom_ind)
        bonded = set()
        for i in ind:
            bonded.update(self.get_bonded_atoms(i))
        return list(bonded)

# This following functions are devoted to figuring out the connectivities and distances.
# Following the conventional approach of setting bond length cutoffs.

_get = lambda l, ind: [l[i] for i in ind]

def _connect(graph, i, j):
    graph[i][j] = 1
    graph[j][i] = 1

def get_bond_cutoff(a1: str, a2: str) -> float:

    """
    Not sure if this works correctly for Cl and Br.
    Needs a sanity check on some complexes.
    Probably works correctly for S.
    """

    a1, a2 = sorted([a1, a2])
    r1, r2 = covalent_radius(a1), covalent_radius(a2)
    cutoff = 1.15 * (r1 + r2)
    if a1 == 'C' and a2 != 'H':
        cutoff = min(2.75, cutoff)
    if a1 == 'H' and ismetal(a2):
        cutoff = 1.1 * (r1 + r2)
    if a2 == 'H' and ismetal(a1):
        cutoff = 1.1 * (r1 + r2)
    # Strict cutoff for Iodine
    if a1 == 'I' and a2 == 'I':
        cutoff = 3
    elif a1 == 'I' or a2 == 'I':
        cutoff = 0.95 * (r1 + r2)
    return cutoff

def get_cutoffs(atoms: list) -> defaultdict:

    """
    Calculate bond length cutoffs for all possible atom pairs.
    """

    unique = list(set(atoms))
    cutoffs = defaultdict(dict)
    n = len(unique)
    for i in range(n):
        for j in range(i, n):
            a1, a2 = unique[i], unique[j]
            cutoff = get_bond_cutoff(a1, a2)
            cutoffs[a1][a2] = cutoff
            cutoffs[a2][a1] = cutoff
    return cutoffs

def get_graph_full_scope(mol: simple_mol) -> np.ndarray:

    """
    If the indices of the ligand atoms are not available, use this function.
    """

    flatten = lambda l: [item for sublist in l for item in sublist]
    n = mol.natoms
    graph = np.zeros((n, n))
    cutoffs = get_cutoffs(mol.atoms)

    matrix = distance_matrix(mol.coords_all, mol.coords_all)

    for i in range(n):
        for j in range(i+1, n):
            atom_i, atom_j = mol.atoms[i], mol.atoms[j]
            if matrix[i][j] <= cutoffs[atom_i][atom_j]:
                _connect(graph, i, j)
    return graph

def get_graph_by_ligands(mol: simple_mol) -> np.ndarray:

    """
    This takes advantage of the fact that the indices of the atoms in each ligand are separated from those of any other ligand.
    Accommodates complexes with any number of metal centers.
    """

    flatten = lambda l: [item for sublist in l for item in sublist]
    n = mol.natoms
    graph = np.zeros((n, n))
    cutoffs = get_cutoffs(mol.atoms)

    mcs = mol.mcs
    lcs = flatten(mol.lcs)
    matrix = distance_matrix(mol.coords_all[mcs], mol.coords_all[lcs])

    for i, ind_i in enumerate(mcs):
        for j, ind_j in enumerate(lcs):
            atom_i, atom_j = mol.atoms[ind_i], mol.atoms[ind_j]
            if matrix[i][j] <= cutoffs[atom_i][atom_j]:
                _connect(graph, ind_i, ind_j)
    
    for i in range(len(mol.ligand_ind)):
        ligand_ind = mol.ligand_ind[i]
        ligand_coords = mol.coords_all[ligand_ind]
        matrix = distance_matrix(ligand_coords, ligand_coords)
        ligand_atoms = _get(mol.atoms, ligand_ind)
        l = len(ligand_atoms)
        for j in range(l):
            for k in range(j+1, l):
                if matrix[j][k] <= cutoffs[ligand_atoms[j]][ligand_atoms[k]]:
                    _connect(graph, ligand_ind[j], ligand_ind[k])
    return graph

def bfs_distances(mol: simple_mol, origin: int, depth: int):

    """
    A breadth-first search algorithm to find the shortest-path distances between any two atoms.
    Only searches for distances up to the given depth.
    """

    all_active = set([origin])
    current_active = set([origin])

    for distance in range(1, depth+1):
        new_active = set(mol.get_bonded_atoms_multiple(list(current_active)))
        new_active -= all_active
        if distance > 1:
            for atom in new_active:
                mol.distances[origin][atom] = distance
        all_active.update(new_active)
        current_active = new_active

def bfs_ligands(mol: simple_mol):

    """
    A breadth-first search algorithm to find out which atoms belong to the same ligand.
    """

    mol.lcs = []
    mol.ligand_ind = []

    graph_copy = mol.graph.copy()
    tmp_lcs = set(mol.get_bonded_atoms_multiple(mol.mcs))
    lcs = []
    
    for i in mol.mcs:
        mol.graph[i] = 0
        mol.graph[:, i] = 0

    while tmp_lcs:
        lc = list(tmp_lcs)[0]
        ligand = set([lc])
        _next = set(mol.get_bonded_atoms(lc)) - ligand
        while _next:
            ligand.update(_next)
            _next = set(mol.get_bonded_atoms_multiple(_next)) - ligand
        lc = list(ligand & tmp_lcs)
        tmp_lcs -= set(lc)
        mol.lcs.append(lc)
        mol.ligand_ind.append(list(ligand))
    
    mol.graph = graph_copy

def determine_CN_NN(mol: simple_mol):

    """
    Determines which ligands are CN/NN.

    Example:
    mol.CN = [0, 1]
    mol.NN = [0]
    """

    mol.CN = []
    mol.NN = []

    for i, lc in enumerate(mol.lcs):
        if sorted([mol.atoms[lc[0]], mol.atoms[lc[1]]]) == ['C', 'N']:
            mol.CN.append(i)
        else:
            mol.NN.append(i)

def get_mol(filename: str, with_ind=True, depth=5) -> simple_mol:

    """
    Creates a simple_mol object from a xyz file.
    """

    mol = simple_mol(filename)
    if with_ind:
        mol.parse_with_ind()
    else:
        mol.parse_all()
    mol.get_all_distances(depth)
    return mol