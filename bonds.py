# This script is solely devoted to figuring out the connectivities.
# Following the conventional approach of setting bond length cutoffs.

from .elements import *
from .molecule import *
from collections import defaultdict
from scipy.spatial import distance_matrix
import numpy as np

get_ = lambda l, ind: [l[i] for i in ind]

def connect(graph, i, j):
    graph[i][j] = 1
    graph[j][i] = 1

def get_bond_cutoff(a1: str, a2: str) -> float:
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

def get_graph_by_ligands(mol: simple_mol) -> np.ndarray:

    """
    This takes advantage of the fact that the indices of the atoms in each ligand are separated from those of any other ligand.
    Assuming only one metal center.
    """

    flatten = lambda l: [item for sublist in l for item in sublist]
    matrix_ = lambda x: distance_matrix(x, x)
    n = len(mol.atoms)
    graph = np.zeros((n, n))
    for j in flatten(mol.lcs):
        connect(graph, mol.mc[0], j)
    
    cutoffs = get_cutoffs(mol.atoms)
    
    for i in range(len(mol.ligand_ind)):
        ligand_ind = mol.ligand_ind[i]
        ligand_coords = mol.coords_all[ligand_ind]
        matrix = matrix_(ligand_coords)
        ligand_atoms = get_(mol.atoms, ligand_ind)
        l = len(ligand_atoms)
        for j in range(l):
            for k in range(j+1, l):
                if matrix[j][k] <= cutoffs[ligand_atoms[j]][ligand_atoms[k]]:
                    connect(graph, ligand_ind[j], ligand_ind[k])
    return graph

def bfs(mol: simple_mol, origin: int, depth: int):

    """
    A breadth-first search algorithms to find the shortest-path distances between any two atoms.
    Only search for distances up to the given depth.
    """

    all_active = set([origin])
    current_active = set([origin])
    for distance in range(1, depth+1):
        new_active = set([])
        for atom in current_active:
            new_active.update(mol.get_bonded_atoms(atom))
        new_active -= all_active
        if distance > 1:
            for atom in new_active:
                mol.distances[origin][atom] = distance
        all_active.update(new_active)
        current_active = new_active