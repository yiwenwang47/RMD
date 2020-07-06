# Revised autocorrelation descriptors. Based on graph of connectivities between atoms.
# Assuming the simple_mol object already has attributes graph and distances.
from .elements import *
from .molecule import *
import numpy as np

operations = {
    'add': lambda x, y: np.add(x, y),
    'subtract': lambda x, y: np.add(x, -y),
    'multiply': lambda x, y: np.multiply(x, y),
    'divide': lambda x, y: np.divide(x, y)
}

def pair_correlation(_property: str, atom1: str, atom2: str, operation: str):
    p1, p2 = properties[_property](atom1), properties[_property](atom2)
    return operations[operation](p1, p2)

def topology_ac(mol: simple_mol, ind1: int, ind2: int, operation: str):
    return operations[operation](mol.coordination[ind1], mol.coordination[ind2])

def rac_from_atom(mol: simple_mol, _property: str, origin: int, scope: set, operation='multiply', depth=3, count=True) -> np.array:
    
    """
    Limiting the first atom in any atom pair to the given origin atom. 
    In other words, this is intened for RACs starting from lc or mc.
    Pass scope = set([]) to get a full-scope rac feature.
    """

    feature = np.zeros(depth+1).astype(np.float)
    
    d_from_origin = mol.distances[origin]
    atom1 = mol.atoms[origin]
    if _property != 'topology':
        feature[0] = pair_correlation(_property, atom1, atom1, operation)
    else:
        feature[0] = topology_ac(mol, origin, origin, operation)

    for d in range(1, depth+1):
        n_d = 0
        targets = set(np.where(d_from_origin==d)[0])
        if scope:
            targets = targets & scope
        for target in targets:
            atom2 = mol.atoms[target]
            if _property != 'topology':
                feature[d] += pair_correlation(_property, atom1, atom2, operation)
            else:
                feature[d] += topology_ac(mol, origin, target, operation)
            n_d += 1
        if count and n_d > 0:
            feature[d] = np.divide(feature[d], n_d)
    
    return feature

def rac_all_atoms(mol:  simple_mol, _property: str, scope: set, operation='multiply', depth=3, count=True) -> np.array:
     
    """
    Does not only start from any specific center.
    Pass scope = set([]) to get a full-scope rac feature.
    """

    if not scope:
        scope = set(range(mol.natoms))
    
    feature = np.zeros(depth+1).astype(np.float)

    for ind in scope:
        atom = mol.atoms[ind]
        if _property != 'topology':
            feature[0] = pair_correlation(_property, atom, atom, operation)
        else:
            feature[0] = topology_ac(mol, ind, ind, operation)

    for d in range(1, depth+1):
        n_d = 0
        targets = np.where(mol.distances==d)
        for i in range(len(targets[0])):
            ind1, ind2 = targets[0][i], targets[1][i]
            if ind1 in scope and ind2 in scope:
                atom1, atom2 = mol.atoms[ind1], mol.atoms[ind2]
                if _property != 'topology':
                    feature[d] += pair_correlation(_property, atom1, atom2, operation)
                else:
                    feature[d] += topology_ac(mol, ind1, ind2, operation)
                n_d += 1
        if count and n_d > 0:
            feature[d] = np.divide(feature[d], n_d)
        if not count:
            feature[d] = np.divide(feature[d], 2) #because any atom pair (i,j) is counted twice

    return feature