# Revised autocorrelation descriptors. Based on graph of connectivities between atoms.
# Assuming the simple_mol object already has attributes graph and distances.

from .elements import *
from .molecule import *
from .utils import *
import numpy as np

def feature_name(start, scope, _property, operation, depth):

    """
    Gives feature name based on all specifications. 
    Intended to be self-explanatory as possible. 
    """

    return '_'.join([start, scope, property_notation[_property], operation_name[operation], str(depth)])

def init_feature(num_properties, operation, depth):

    """
    Initializes an empty feature vector. 
    """

    if operation == 'subtract' or operation == 'divide':
        depth -= 1
    return np.zeros((depth+1) * num_properties).astype(np.float)

def pair_correlation(_property: str, atom1: str, atom2: str, operation: str):
    p1, p2 = properties[_property](atom1), properties[_property](atom2)
    return operations[operation](p1, p2)

def topology_ac(mol: simple_mol, ind1: int, ind2: int, operation: str):
    return operations[operation](mol.coordination[ind1], mol.coordination[ind2])

def RAC_from_atom(mol: simple_mol, _property: str, origin: int, scope: set, operation: str, depth: int, average: bool) -> np.ndarray:
    
    """
    Limiting the first atom in any atom pair to the given origin atom. 
    In other words, this is intended for RACs starting from lc or mc.
    Pass scope = set() to get a full-scope rac feature.
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
        if average and n_d > 0:
            feature[d] = np.divide(feature[d], n_d)

    if operation == 'subtract' or operation == 'divide':
        return feature[1:] #Because zero depth is trivial
    
    return feature

def RAC_all_atoms(mol: simple_mol, _property: str, scope: set, operation: str, depth: int, average: bool) -> np.ndarray:
     
    """
    Does not only start from any specific center.
    Pass scope = set() to get a full-scope rac feature.
    """
    
    feature = np.zeros(depth+1).astype(np.float)

    for ind in range(mol.natoms):
        atom = mol.atoms[ind]
        if _property != 'topology':
            feature[0] += pair_correlation(_property, atom, atom, operation)
        else:
            feature[0] += topology_ac(mol, ind, ind, operation)

    if average:
        if not scope:
            feature[0] = np.divide(feature[0], mol.natoms)
        else:
            feature[0] = np.divide(feature[0], len(scope))

    for d in range(1, depth+1):
        n_d = 0
        targets = np.where(mol.distances==d)
        for i in range(len(targets[0])):
            ind1, ind2 = targets[0][i], targets[1][i]
            if not scope or (ind1 in scope and ind2 in scope):
                atom1, atom2 = mol.atoms[ind1], mol.atoms[ind2]
                if _property != 'topology':
                    feature[d] += pair_correlation(_property, atom1, atom2, operation)
                else:
                    feature[d] += topology_ac(mol, ind1, ind2, operation)
                n_d += 1
        if average and n_d > 0:
            feature[d] = np.divide(feature[d], n_d)
        if not average:
            feature[d] = np.divide(feature[d], 2) #because any atom pair (i,j) is counted twice

    if operation == 'subtract' or operation == 'divide':
        return feature[1:] #because zero depth is trivial

    return feature

# Set to False for now. Need more experiments.
_average = {
    'electronegativity': False,
    'atomic number': False,
    'identity': False,
    'covalent radius': False,
    'topology': False,
    'polarizability': False
}

def multiple_RACs_from_atom(mol: simple_mol, _properties: list, origin: int, scope: set, operation: str, depth: int) -> np.ndarray:
    for i, _property in enumerate(_properties):
        _new = RAC_from_atom(mol, _property=_property, origin=origin, scope=scope, operation=operation, depth=depth, average=_average[_property])
        if i == 0:
            feature = _new
        else:
            feature = np.concatenate((feature, _new))
    return feature

def multiple_RACs_all_atoms(mol: simple_mol, _properties: list, scope: set, operation: str, depth: int) -> np.ndarray:
    for i, _property in enumerate(_properties):
        _new = RAC_all_atoms(mol, _property=_property, scope=scope, operation=operation, depth=depth, average=_average[_property])
        if i == 0:
            feature = _new
        else:
            feature = np.concatenate((feature, _new))
    return feature

def RAC_f_all(mol: simple_mol, _properties: list, operation: str, depth: int) -> np.ndarray:
    return multiple_RACs_all_atoms(mol=mol, _properties=_properties, scope=set(), operation=operation, depth=depth)

def RAC_mc_all(mol: simple_mol, _properties: list, operation: str, depth: int, average_mc=True) -> np.ndarray:

    """
    The feature vector is averaged over all metal centers by default.
    """

    n_mc = len(mol.mcs)
    assert n_mc > 0

    feature = init_feature(len(_properties), operation, depth)

    for i, mc in enumerate(mol.mcs):
        feature += multiple_RACs_from_atom(mol, _properties=_properties, origin=mc, scope=set(), operation=operation, depth=depth)
    
    if not average_mc:
        return feature

    return np.divide(feature, n_mc)

def RAC_lc_ligand(mol: simple_mol, ligand_type: str, _properties: list, operation: str, depth: int, average_lc=True) -> np.ndarray:

    """
    The ligand type is not specified here in order to accommodate more possibilities in the future.
    Within each ligand, the feature vector is averaged over all ligand centers by default.
    Averaged over all ligands. 
    """

    ligands = mol.get_specific_ligand(ligand_type)
    feature = init_feature(len(_properties), operation, depth)

    for ligand in ligands:
        ligand_feature = init_feature(len(_properties), operation, depth)
        lcs = mol.lcs[ligand]
        scope = set(mol.ligand_ind[ligand])
        for lc in lcs:
            ligand_feature += multiple_RACs_from_atom(mol, _properties=_properties, origin=lc, scope=scope, operation=operation, depth=depth)
        ligand_feature = np.divide(ligand_feature, len(lcs))
        feature += ligand_feature

    if not average_lc:
        return feature

    return np.divide(feature, len(ligands))

def RAC_f_ligand(mol: simple_mol, ligand_type: str, _properties: list, operation: str, depth: int) -> np.ndarray:

    """
    The ligand type is not specified here in order to accommodate more possibilities in the future.
    Averaged over all ligands.
    """

    ligands = mol.get_specific_ligand(ligand_type)
    feature = init_feature(len(_properties), operation, depth)

    for ligand in ligands:
        scope = set(mol.ligand_ind[ligand])
        feature += multiple_RACs_all_atoms(mol, _properties=_properties, scope=scope, operation=operation, depth=depth)
    
    return np.divide(feature, len(ligands))

#The following section is only about RAC features for CN/NN ligands. Still following the RAC-155 list.

def RAC_names_CN_NN(depth=3) -> list:

    """
    Get all the feature names.
    """

    _properties = ['electronegativity', 'atomic number', 'identity', 'covalent radius', 'topology']

    names = []

    def helper(start, scope, operation):
        _new = []
        for _property in _properties:
            if operation == 'multiply' or operation == 'add':
                _new += [feature_name(start, scope, _property, operation, d) for d in range(depth+1)]
            else:
                if _property != 'identity':
                    _new += [feature_name(start, scope, _property, operation, d) for d in range(1, depth+1)]
        return _new

    names += helper('f', 'all', 'multiply')
    names += helper('mc', 'all', 'multiply')
    names += helper('lc', 'CN', 'multiply')
    names += helper('lc', 'NN', 'multiply')
    names += helper('f', 'CN', 'multiply')
    names += helper('f', 'NN', 'multiply')

    names += helper('mc', 'all', 'subtract')
    names += helper('lc', 'CN', 'subtract')
    names += helper('lc', 'NN', 'subtract')

    return names

def RAC_graph(names: list) -> simple_graph:

    """
    Grouping the RAC features.
    """

    translation = {
        '0': 'proximal',
        '1': 'proximal',
        '2': 'middle',
        '3': 'distal'
    }

    return create_feature_graph(names, translation=translation)

def full_RAC_CN_NN(mol: simple_mol, depth=3) -> np.ndarray:

    """
    A modified version of the original RAC-155.
    Discriminating between axial and equatorial ligands is meanningless. In our case, we identify CN and NN ligands instead.
    The start/scope definitions we adopt are:
    f/all, mc/all, lc/CN, lc/NN, f/CN, f/NN for product RACs
    mc/all, lc/CN, lc/NN for difference RACs

    Whenever scope is defined to be one type of ligand, the feature vector is averaged over all corresponding ligands. 

    Also, the original RAC-155 failed to recognize the difference between averaging over all counted atom pairs and not doing so.
    For example, it would not make sense to average 'identity', but we probably should average 'electronegativity'. 
    This is specified above in the dictionary _average.

    According to J.P Janet et al.(2017), some of the features are trivial. But for now, we will not exclude them.

    Unfinished.
    """

    _properties = ['electronegativity', 'atomic number', 'identity', 'covalent radius', 'topology']
    __properties = ['electronegativity', 'atomic number', 'covalent radius', 'topology']

    _f_all = RAC_f_all(mol=mol, _properties=_properties, operation='multiply', depth=depth)
    _mc_all = RAC_mc_all(mol=mol, _properties=_properties, operation='multiply', depth=depth)
    _lc_CN = RAC_lc_ligand(mol=mol, ligand_type='CN', _properties=_properties, operation='multiply', depth=depth, average_lc=True)
    _lc_NN = RAC_lc_ligand(mol=mol, ligand_type='NN', _properties=_properties, operation='multiply', depth=depth, average_lc=True)
    _f_CN = RAC_f_ligand(mol=mol, ligand_type='CN', _properties=_properties, operation='multiply', depth=depth)
    _f_NN = RAC_f_ligand(mol=mol, ligand_type='NN', _properties=_properties, operation='multiply', depth=depth)

    __mc_all = RAC_mc_all(mol=mol, _properties=__properties, operation='subtract', depth=depth)
    __lc_CN = RAC_lc_ligand(mol=mol, ligand_type='CN', _properties=__properties, operation='subtract', depth=depth, average_lc=True)
    __lc_NN = RAC_lc_ligand(mol=mol, ligand_type='NN', _properties=__properties, operation='subtract', depth=depth, average_lc=True)

    full_feature = np.concatenate((_f_all, _mc_all, _lc_CN, _lc_NN, _f_CN, _f_NN, __mc_all, __lc_CN, __lc_NN))

    return full_feature