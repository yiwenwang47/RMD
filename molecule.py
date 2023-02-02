import numpy as np
from .elements import elementdict, covalent_radius, ismetal, properties
from collections import defaultdict
import copy
from scipy.spatial import distance_matrix

class simple_atom(object):

    """
    A very simple atom object. Intended for much more complicated descriptors.
    For now, only special properties will be carried by this object.
    """

    def __init__(self, element: str):
        self.atom = element
        self.atomic_mass = elementdict[element][0]
        self.atomic_number = elementdict[element][1]
        self.covalent_radius = covalent_radius(element)
        self.properties = {}
    
    def __str__(self):
        return self.atom

    def __repr__(self):
        return self.atom
    
    def copy(self):
        return copy.deepcopy(self)

    def add_property(self, name: str, _property: np.float):
        self.properties[name] = _property

class simple_mol(object):

    """
    A very simple molecule object. Parses information from a xyz file. 
    Now also deals with generic cases.
    The coordinating atoms are also listed as lc, which stands for ligand center.
    """

    def __init__(self, complex=True):
        if complex:
            self.complex = True
            self.ligand_types = [] #could be determined later by any custom functions, must be in the same order as self.lcs and self.ligand_ind
        self.properties = {}
    
    def from_file(self, filename: str):
        f = open(filename, 'r')
        self.text = f.read().split('\n')

    def __repr__(self):
        return ', '.join(self.atoms)

    def copy(self):
        return copy.deepcopy(self)

    def get_coords(self):

        """
        Parse the coordinates from an xyz file.
        """

        coord_lines = []
        for line in self.text:
            split = line.split()
            if len(split) == 4:
                coord_lines.append(split)
        def helper(split):
            atom = split[0]
            coords = np.array([float(split[i]) for i in range(1,4)])
            return atom, coords
        atom, coords = helper(coord_lines[0])
        atoms = [atom]
        coords_all = np.array([coords])
        if self.complex:
            self.mcs = []
            if ismetal(atom):
                self.mcs.append(0)
        for i in range(1, len(coord_lines)):
            atom, coords = helper(coord_lines[i])
            atoms.append(atom)
            if self.complex and ismetal(atom):
                self.mcs.append(i)
            coords_all = np.concatenate((coords_all, np.array([coords])), axis=0)
        self.atoms = atoms
        self.natoms = len(atoms)
        self.coords_all = coords_all
    
    def create_dict_graph(self):
        self.dict_graph = {}
        for i in range(self.natoms):
            self.dict_graph[i] = list(np.where(self.graph[i]==1)[0])

    def get_special_graph_copy(self):

        """
        A graph cheat. Removes the mcs.
        Only to be called in get_cycles()
        """

        copy = self.dict_graph.copy()
        for mc in self.mcs:
            copy.pop(mc, None)
        for lcs in self.lcs:
            for i in lcs:
                copy[i] = list(set(copy[i])-set(self.mcs))
        return copy

    def init_distances(self):
        self.distances = self.graph.copy()

    def distance_cheat(self, fake_depth: int):

        """
        This is a trick that is helpful when we only care about distances up to a given depth, for example, 3. 
        And then we will treat all the distances greater than 3 the same as fake_depth. 
        """

        self.distances[np.where(self.distances==0)] = fake_depth
        self.distances[np.arange(self.natoms), np.arange(self.natoms)] = 0

    def get_all_distances(self, depth: int, fake_depth=0):

        """
        Only calculates shortest-path distances up to depth.
        New option: set all the longer distances as fake_depth.
        """

        for ind in range(self.natoms):
            bfs_distances(self, ind, depth)
        if fake_depth != 0:
            self.distance_cheat(fake_depth)
    
    def parse_all(self, complex=True):

        """
        A more general case.
        """

        self.get_coords()
        del self.text
        self.graph = get_graph_full_scope(self)
        if complex:
            self.complex = True
            self.lcs = []
            self.ligand_ind = []
            bfs_ligands(self)
        self.custom_property('topology', self.graph.sum(axis=0))
        self.init_distances()

    def get_bonded_atoms(self, atom_index: int):
        con = self.graph[atom_index]
        return np.where(con==1)[0]

    def get_bonded_atoms_multiple(self, atom_ind: list) -> list:
        indices = set(atom_ind)
        bonded = set()
        for i in indices:
            bonded.update(self.get_bonded_atoms(i))
        return list(bonded)

    def get_specific_ligand(self, ligand_type: str) -> list:

        """
        Returns a list of ligands that belong to the given type.
        For example, 
        [0,1]
        means the first and second ligands are the given type.
        This could be used to access self.lcs and self.ligand_ind to get the indices of ligand centers and all ligand atoms.
        """

        return [i for i, x in enumerate(self.ligand_types) if x == ligand_type]

    def populate_property(self, _property: str):
        if _property in properties and _property not in self.properties:
            self.properties[_property] = np.zeros(self.natoms)
            _func = properties[_property]
            for i in range(self.natoms):
                self.properties[_property][i] = _func(self.atoms[i])

    def custom_property(self, _property: str, values):
        
        """
        Apart from elemental properties, each atom could have properties from other calculations. 
        """

        self.properties[_property] = values

# These following functions are devoted to figuring out the connectivities and distances.
# Following the conventional approach of setting bond length cutoffs.

_get = lambda l, indices: [l[i] for i in indices]

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
    if a1 == 'C' and a2 == 'Cl':
        cutoff = 2.1
    if a1 == 'H' and ismetal(a2):
        cutoff = 1.1 * (r1 + r2)
    if a2 == 'H' and ismetal(a1):
        cutoff = 1.1 * (r1 + r2)
    # Strict cutoff for Iodine
    if a1 == 'I' and a2 == 'I':
        cutoff = 3
    if a1 == 'Ir' and a2 == 'N':
        cutoff = 2.5
    # elif a1 == 'I' or a2 == 'I':
    #     cutoff = 0.95 * (r1 + r2)
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

    n = mol.natoms
    graph = np.zeros((n, n))
    cutoffs = get_cutoffs(mol.atoms)

    mol.matrix = distance_matrix(mol.coords_all, mol.coords_all)

    for i in range(n):
        for j in range(i+1, n):
            atom_i, atom_j = mol.atoms[i], mol.atoms[j]
            if mol.matrix[i][j] <= cutoffs[atom_i][atom_j]:
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

    """
    This part connects the metal atoms and the coordinating atoms in the ligands first. 
    """

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
        if not new_active:
            break
        if distance > 1:
            for atom in new_active:
                mol.distances[origin][atom] = distance
        all_active.update(new_active)
        current_active = new_active

def bfs_ligands(mol: simple_mol):

    """
    A breadth-first search algorithm to find out which atoms belong to the same ligand.
    """

    graph_copy = mol.graph.copy()
    tmp_lcs = set(mol.get_bonded_atoms_multiple(mol.mcs))
    
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

def get_cycles(graph: dict, origin: int, verbose=False) -> list:

    """
    A very simple dfs algorithm to find cycles in a graph.
    Origin does not have to be an integer.
    """

    visited = set()
    current = []
    cycles = []
    def dfs(visited, current, graph, node, cycles):
        if node not in visited:
            if verbose:
                print(str(node) + ' is visited')
            visited.add(node)
            current.append(node)
            if set(graph[node])&visited:
                i = len(current) - 2
                if i >= 0:
                    for neighbor in set(graph[node])&visited:
                        j = min(current.index(neighbor), i)
                        if j < i:
                                cycles.append(current[j:])
            if set(graph[node]).issubset(visited):
                current = current[:-1]
            else:
                for neighbour in graph[node]:
                    dfs(visited, current.copy(), graph, neighbour, cycles)
    dfs(visited, current, graph, origin, cycles)
    return cycles

def get_cycles_molecule(mol: simple_mol) -> list:
    
    """
    Returns a list of all the cycles in the ligands.
    """

    copy = mol.get_special_graph_copy()
    cycles = []
    for lc in mol.lcs:
        cycles += get_cycles(copy, lc[0], verbose=False)
    return cycles

def get_mol(filename: str, depth=5, fake_depth=0) -> simple_mol:

    """
    Creates a simple_mol object from a xyz file.
    """

    mol = simple_mol()
    mol.from_file(filename)
    mol.parse_all()
    mol.get_all_distances(depth, fake_depth=fake_depth)
    return mol