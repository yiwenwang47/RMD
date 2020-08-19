# Revised autocorrelation descriptors based on the topology of a molecule.
# For the sake of simplicity, type hints simple_mol are omitted. 
# Any variable named mol should be a simple_mol object, and have the following attributes: graph and distances.

from .elements import property_notation, operation_name, operations
from .utils import simple_graph, create_feature_graph
import numpy as np

class StyleError(Exception):

    """
    Exception raised when an incorrect autocorrelation style is given.
    """

    def __init__(self, style, message="is not any of the allowed styles: Moreau-Broto, Moran or Geary"):
        self.style = style
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f'{self.style} {self.message}'

delta = lambda x, y: np.int64(x==y)

def feature_name(start, scope, _property, operation, d):

    """
    Gives feature name or RAC based on all specifications. 
    Intended to be as self-explanatory as possible. 
    """

    return '_'.join([start, scope, property_notation[_property], operation_name[operation], str(d)])

def init_feature(num_properties, operation, d):

    """
    Initializes an empty feature vector. 
    """

    if operation == 'subtract' or operation == 'divide':
        d -= 1
    return np.zeros((d+1) * num_properties).astype(np.float)

# This function is written in a very naive way. Will only be called in special cases.
def property_correlation(mol, _property: str, ind1: int, ind2: int, operation: str) -> np.float:
    assert _property in mol.properties
    p1, p2 = mol.properties[_property][ind1],  mol.properties[_property][ind2]
    return operations[operation](p1, p2)

# Moreau-Broto autocorrelation calculated by matrix multiplication.
def Moreau_Broto_ac(array_1: np.ndarray, binary_matrix: np.ndarray, array_2: np.ndarray, operation: str, with_origin=False, cross_scope=False) -> np.float:

    """
    Although the original Moreau-Broto autocorrelation only considers product aurtocorrelation, I try to accommodate all four operations.
    By default, the binary_matrix is assumed to be 2-d and symmetric. If with_origin = True, binary_matrix should be a 1-d array.
    This does not deal with potential problems brought by zero-ish elemental property values.
    """

    if not with_origin:
        assert len(binary_matrix.shape) == 2
        if cross_scope:
            assert binary_matrix.shape[0] == len(array_1) and binary_matrix.shape[1] == len(array_2)
        if operation == 'divide': 
            array_2 = 1/array_2
            operation = 'multiply'
        if operation == 'add' or operation == 'subtract':
            array_1 = binary_matrix.sum(axis=1) * array_1
        array_2 = binary_matrix.dot(array_2)
        return operations[operation](array_1, array_2).sum()
    else:
        assert len(binary_matrix.shape) == 1
        return (operations[operation](array_1, array_2) * binary_matrix).sum()

# Autocorrelation inspired by Moran's I
def Moran_ac(array_1: np.ndarray, binary_matrix: np.ndarray, array_2: np.ndarray, denominator: np.float, mean: np.float, with_origin=False, cross_scope=False) -> np.float:

    """
    The denominator and mean are left as parameters in order to allow revised versions. 
    """

    assert not (with_origin and cross_scope)
    denominator = np.int64(binary_matrix>0).sum() * denominator
    if denominator == 0:
        return np.float(0)
    array_1, array_2 = array_1 - mean, array_2 - mean
    if with_origin:
        assert len(binary_matrix.shape) == 1
        return (binary_matrix * array_2).dot(array_1) / denominator 
    else:
        if cross_scope:
            assert binary_matrix.shape[0] == len(array_1) and binary_matrix.shape[1] == len(array_2)
        else:
            assert len(binary_matrix.shape) == 2
        return array_1.dot(binary_matrix).dot(array_2) / denominator 

# Autocorrelation inspired by Geary's C
def Geary_ac(array_1: np.ndarray, binary_matrix: np.ndarray, array_2: np.ndarray, denominator: np.float, with_origin=False, cross_scope=False) -> np.float:

    """
    The denominator is left as a parameter in order to allow revised versions.
    If with_origin=False, array_2 is assumed to be identical to array_1.
    """

    assert not (with_origin and cross_scope)
    binary_matrix_copy = np.int64(binary_matrix>0)
    denominator = 2 * binary_matrix_copy.sum() * denominator
    if denominator == 0:
        return np.float(0)
    if with_origin:
        assert len(binary_matrix.shape) == 1
        array = array_1 - array_2
        result = array * binary_matrix * array
    else:
        if cross_scope:
            assert binary_matrix.shape[0] == len(array_1) and binary_matrix.shape[1] == len(array_2)
        else:
            assert len(binary_matrix.shape) == 2
            assert ((array_1-array_2)**2).sum() < 1e-5
        s = binary_matrix * array_2
        s_copy = binary_matrix_copy * array_2
        # number of x_j * x_i ^ 2 - 2 x_i * sum(x_j) + sum(x_j ^ 2)
        result = binary_matrix.sum(axis=1) * array_1 * array_1 - 2 * array_1 * s.sum(axis=1) + (s * s_copy).sum(axis=1)
    return result.sum() / denominator

# This section includes the Moran-styled and Geary-styled autocorrelation functions.
# Although this might seem repetitive, they need to be written separately because unlike Moreau-Broto AC,
# they couldn't be done by any operation other than multiplication.

def AC_from_atom(mol, style: str, _property: str, origin: int, scope: set, depth: tuple, three_d=False) -> np.ndarray:

    """
    Style: 'Moran' or 'Geary'
    The parameter scope is forced to be a set just to prevent possible replicates. Pass scope = set() to get a full-scope feature.
    """

    assert depth[0] != 0 # because it's trivial

    mol.populate_property(_property)
    _length = depth[1] - depth[0] + 1
    feature = np.zeros(_length).astype(np.float)
    origin_property = mol.properties[_property][origin]
    d_from_origin = mol.distances[origin].copy()
    array_2 = mol.properties[_property].copy()
    if three_d:
        d_from_origin_3d = mol.matrix[origin].copy()
    if scope:
        scope = list(scope)
        d_from_origin = d_from_origin[scope]
        if three_d:
            d_from_origin_3d = d_from_origin_3d[scope]
        array_2 = array_2[scope]
    array_1 = origin_property * np.ones(len(d_from_origin))

    denominator = (array_2.std())**2
    if denominator == 0:
        return feature
    if style == 'Geary':
        denominator = denominator * len(array_2) / (len(array_2)-1)
    if style == 'Moran':
        mean = array_2.mean()

    for i in range(_length):
        d = depth[0] + i
        targets = delta(d_from_origin, d)
        if three_d:
            targets = targets * d_from_origin_3d
        if style == 'Moran':
            feature[i] = Moran_ac(array_1, targets, array_2, denominator=denominator, mean=mean, with_origin=True)
        else:
            feature[i] = Geary_ac(array_1, targets, array_2, denominator=denominator, with_origin=True)
    
    return feature

def AC_all_atoms(mol, style: str, _property: str, scope: set, depth: tuple, three_d=False) -> np.ndarray:

    """
    Style: 'Moran' or 'Geary'
    The parameter scope is forced to be a set just to prevent possible replicates. Pass scope = set() to get a full-scope feature.
    """

    assert depth[0] != 0 # because it's trivial
    
    mol.populate_property(_property)
    _length = depth[1] - depth[0] + 1
    feature = np.zeros(_length).astype(np.float)
    array = mol.properties[_property].copy()
    distances = mol.distances.copy()
    if three_d:
        matrix = mol.matrix.copy()
    if scope:
        scope = list(scope)
        array = array[scope]
        distances = distances[scope][:, scope]
        if three_d:
            matrix = matrix[scope][:, scope]
    
    denominator = (array.std())**2
    if denominator == 0:
        return feature
    if style == 'Geary':
        denominator = denominator * len(array) / (len(array)-1)
    if style == 'Moran':
        mean = array.mean()

    for i in range(_length):
        d = depth[0] + i
        targets = delta(distances, d)
        if three_d:
            targets = targets * matrix
        if style == 'Moran':
            feature[i] = Moran_ac(array, targets, array, denominator=denominator, mean=mean, with_origin=False)
        else:
            feature[i] = Geary_ac(array, targets, array, denominator=denominator, with_origin=False)
    
    return feature

def AC_cross_scope(mol, style: str, _property: str, scope_1: set, scope_2: set, depth: tuple, three_d=False) -> np.ndarray:

    """
    Style: 'Moran' or 'Geary'
    Cross-scope autocorrelation descriptors. For any valid atom pair, the first atom belongs to scope_1, and the second belongs to scope_2.
    The parameters scope_1 and scope_2 are forced to be sets just to prevent possible replicates.
    """

    assert depth[0] != 0 # because it's trivial
    assert len(scope_1) > 0 and len(scope_2) > 0
    
    mol.populate_property(_property)
    _length = depth[1] - depth[0] + 1
    feature = np.zeros(_length).astype(np.float)
    array = mol.properties[_property].copy()
    distances = mol.distances.copy()
    if three_d:
        matrix = mol.matrix.copy()
    scope_1, scope_2, scope = list(scope_1), list(scope_2), list(scope_1|scope_2)

    array_copy = array[scope]
    denominator = (array_copy.std())**2
    if denominator == 0:
        return feature
    if style == 'Geary':
        denominator = denominator * len(array_copy) / (len(array_copy)-1)
    if style == 'Moran':
        mean = array_copy.mean()
    
    array_1, array_2 = array[scope_1], array[scope_2]
    distances = distances[scope_1][:, scope_2]
    if three_d:
        matrix = matrix[scope_1][:, scope_2]

    for i in range(_length):
        d = depth[0] + i
        targets = delta(distances, d)
        if three_d:
            targets = targets * matrix
        if style == 'Moran':
            feature[i] = Moran_ac(array_1, targets, array_2, denominator=denominator, mean=mean, cross_scope=True)
        else:
            feature[i] = Geary_ac(array_1, targets, array_2, denominator=denominator, cross_scope=True)

    return feature

# This section includes revised Moreau-Broto autocorrelation functions.

def MB_from_atom(mol, _property: str, origin: int, scope: set, operation: str, depth: tuple, average: bool, three_d=False) -> np.ndarray:
    
    """
    Limiting the first atom in any atom pair to the given origin atom. 
    In other words, this is intended for RACs starting from lc or mc.
    The parameter scope is forced to be a set just to prevent possible replicates. Pass scope = set() to get a full-scope rac feature.
    """

    mol.populate_property(_property)
    _length = depth[1] - depth[0] + 1
    feature = np.zeros(_length).astype(np.float)
    d_from_origin = mol.distances[origin].copy()
    if three_d:
        d_from_origin_3d = mol.matrix[origin].copy()
    array_2 = mol.properties[_property].copy()
    if scope:
        scope = list(scope)
        d_from_origin = d_from_origin[scope]
        if three_d:
            d_from_origin_3d = d_from_origin_3d[scope]
        array_2 = array_2[scope]
    array_1 = mol.properties[_property][origin] * np.ones(len(d_from_origin))

    if depth[0] == 0:
        feature[0] = property_correlation(mol, _property, origin, origin, operation)
        a, b = 1, _length
    else:
        a, b = 0, _length

    for i in range(a, b):
        d = depth[0] + i
        targets = delta(d_from_origin, d)
        n_d = targets.sum()
        if three_d:
            targets = targets * d_from_origin_3d
        feature[i] = Moreau_Broto_ac(array_1, targets, array_2, operation, with_origin=True)
        if average and n_d > 0:
            feature[i] = np.divide(feature[i], n_d)
    
    if operation == 'subtract' or operation == 'divide':
        return feature[1:] #Because zero depth is trivial

    return feature

def MB_all_atoms(mol, _property: str, scope: set, operation: str, depth: tuple, average: bool, three_d=False) -> np.ndarray:
     
    """
    Does not only start from any specific center.
    The parameter scope is forced to be a set just to prevent possible replicates. Pass scope = set() to get a full-scope rac feature.
    """
    
    mol.populate_property(_property)
    _length = depth[1] - depth[0] + 1
    feature = np.zeros(_length).astype(np.float)
    array = mol.properties[_property].copy()
    distances = mol.distances.copy()
    if three_d:
        matrix = mol.matrix.copy()
    n_0 = mol.natoms
    if scope:
        scope = list(scope)
        array, n_0, distances = array[scope], len(scope), distances[scope][:, scope]
        if three_d:
            matrix = matrix[scope][:, scope]
    if depth[0] == 0:
        feature[0] = Moreau_Broto_ac(array, np.ones(n_0), array, operation, with_origin=True)
        if average and n_0 > 0:
            feature[0] = np.divide(feature[0], n_0)
        a, b = 1, _length
    else:
        a, b = 0, _length

    for i in range(a, b):
        d = depth[0] + i
        targets = delta(distances, d)
        n_d = targets.sum()
        if three_d:
            targets = targets * matrix
        if n_d > 0:
            feature[i] = Moreau_Broto_ac(array, targets, array, operation, with_origin=False)
            if average:
                feature[i] = np.divide(feature[i], n_d)

    if operation == 'subtract' or operation == 'divide':
        return feature[1:] #because zero depth is trivial

    return feature

def MB_cross_scope(mol, _property: str, scope_1: set, scope_2: set, operation: str, depth: tuple, average: bool, three_d=False) -> np.ndarray:
    
    """
    Cross-scope autocorrelation descriptors. For any valid atom pair, the first atom belongs to scope_1, and the second belongs to scope_2.
    """
    
    assert depth[0] != 0 #because it's ill-defined

    mol.populate_property(_property)
    _length = depth[1] - depth[0] + 1
    feature = np.zeros(_length).astype(np.float)
    array = mol.properties[_property].copy()
    scope_1, scope_2 = list(scope_1), list(scope_2)
    array_1, array_2 = array[scope_1], array[scope_2]
    distances = mol.distances.copy()[scope_1][:, scope_2]
    if three_d:
        matrix = mol.matrix.copy()[scope_1][:, scope_2]

    for i in range(_length):
        d = depth[0] + i
        targets = delta(distances, d)
        n_d = targets.sum()
        if three_d:
            targets = targets * matrix
        if n_d > 0:
            feature[i] = Moreau_Broto_ac(array_1, targets, array_2, operation, cross_scope=True)
            if average:
                feature[i] = np.divide(feature[i], n_d)
    
    return feature

# Set to False for now. Needs more experiments.
_average = {
    'electronegativity': False,
    'atomic number': False,
    'identity': False,
    'covalent radius': False,
    'topology': False,
    'polarizability': False,
    'vdW radius': False
}

# This section includes higher level RAC functions.
# From this point on, all these RAC functions will support all three styles.

def multiple_RACs_from_atom(mol, _properties: list, origin: int, scope: set, depth: tuple, operation='multiply', style='Moreau-Broto', three_d=False) -> np.ndarray:
    if style not in ['Moreau-Broto', 'Moran', 'Geary']:
        raise StyleError(style)
    for i, _property in enumerate(_properties):
        if style == 'Moreau-Broto':
            if _property in _average:
                average = _average[_property]
            else:
                average = True
            _new = MB_from_atom(mol, _property=_property, origin=origin, scope=scope, operation=operation, depth=depth, average=average, three_d=three_d)
        else:
            _new = AC_from_atom(mol, style=style, _property=_property, origin=origin, scope=scope, depth=depth, three_d=three_d)
        if i == 0:
            feature = _new
        else:
            feature = np.concatenate((feature, _new))
    return feature

def multiple_RACs_all_atoms(mol, _properties: list, scope: set, depth: tuple, operation='multiply', style='Moreau-Broto', three_d=False) -> np.ndarray:
    if style not in ['Moreau-Broto', 'Moran', 'Geary']:
        raise StyleError(style)
    for i, _property in enumerate(_properties):
        if style == 'Moreau-Broto':
            if _property in _average:
                average = _average[_property]
            else:
                average = True
            _new = MB_all_atoms(mol, _property=_property, scope=scope, operation=operation, depth=depth, average=average, three_d=three_d)
        else:
            _new = AC_all_atoms(mol, style=style, _property=_property, scope=scope, depth=depth, three_d=three_d)
        if i == 0:
            feature = _new
        else:
            feature = np.concatenate((feature, _new))
    return feature

def multiple_RACs_cross_scope(mol, _properties: list, scope_1: set, scope_2: set, depth: tuple, operation='multiply', style='Moreau-Broto', three_d=False) -> np.ndarray:
    if style not in ['Moreau-Broto', 'Moran', 'Geary']:
        raise StyleError(style)
    for i, _property in enumerate(_properties):
        if style == 'Moreau-Broto':
            if _property in _average:
                average = _average[_property]
            else:
                average = True
            _new = MB_cross_scope(mol, _property=_property, scope_1=scope_1, scope_2=scope_2, operation=operation, depth=depth, average=average, three_d=three_d)
        else:
            _new = AC_cross_scope(mol, style=style, _property=_property, scope_1=scope_1, scope_2=scope_2, depth=depth, three_d=three_d)
        if i == 0:
            feature = _new
        else:
            feature = np.concatenate((feature, _new))
    return feature

def RAC_f_all(mol, _properties: list, depth: tuple, operation='multiply', style='Moreau-Broto', three_d=False) -> np.ndarray:
    return multiple_RACs_all_atoms(mol=mol, _properties=_properties, scope=set(), depth=depth, operation=operation, style=style, three_d=three_d)

def RAC_mc_all(mol, _properties: list, depth: tuple, operation='multiply', style='Moreau-Broto', average_mc=True, three_d=False) -> np.ndarray:

    """
    The feature vector is averaged over all metal centers by default.
    """

    n_mc = len(mol.mcs)
    assert n_mc > 0
    feature = init_feature(len(_properties), operation, depth[1]-depth[0])
    for mc in mol.mcs:
        feature += multiple_RACs_from_atom(mol, _properties=_properties, origin=mc, scope=set(), depth=depth, operation=operation, style=style, three_d=three_d)
    
    if not average_mc:
        return feature
    return np.divide(feature, n_mc)

def RAC_mc_ligand(mol, ligand_type: str, _properties: list, depth: tuple, operation='multiply', style='Moreau-Broto', average_mc=True, three_d=False) -> np.ndarray:

    """
    The feature vector is averaged over all metal centers and all ligands by default.
    """

    n_mc = len(mol.mcs)
    ligands = mol.get_specific_ligand(ligand_type)
    assert n_mc > 0
    feature = init_feature(len(_properties), operation, depth[1]-depth[0])
    for ligand in ligands:
        for mc in mol.mcs:
            scope = set(mol.ligand_ind[ligand])
            scope.update([mc])
            feature += multiple_RACs_from_atom(mol, _properties=_properties, origin=mc, scope=scope, depth=depth, operation=operation, style=style, three_d=three_d)
    
    if average_mc:
        feature = np.divide(feature, n_mc) 
    return np.divide(feature, len(ligands))

def RAC_lc_ligand(mol, ligand_type: str, _properties: list, depth: tuple, operation='multiply', style='Moreau-Broto', average_lc=True, three_d=False) -> np.ndarray:

    """
    The ligand type is not specified here in order to accommodate more possibilities in the future.
    Within each ligand, the feature vector is averaged over all ligand centers by default.
    Averaged over all ligands. 
    """

    ligands = mol.get_specific_ligand(ligand_type)
    feature = init_feature(len(_properties), operation, depth[1]-depth[0])

    for ligand in ligands:
        ligand_feature = np.zeros(len(feature))
        lcs = mol.lcs[ligand]
        scope = set(mol.ligand_ind[ligand])
        for lc in lcs:
            ligand_feature += multiple_RACs_from_atom(mol, _properties=_properties, origin=lc, scope=scope, depth=depth, operation=operation, style=style, three_d=three_d)
        ligand_feature = np.divide(ligand_feature, len(lcs))
        feature += ligand_feature

    if not average_lc:
        return feature
    return np.divide(feature, len(ligands))

def RAC_f_ligand(mol, ligand_type: str, _properties: list, depth: tuple, operation='multiply', style='Moreau-Broto', three_d=False) -> np.ndarray:

    """
    The ligand type is not specified here in order to accommodate more possibilities in the future.
    Averaged over all ligands.
    """

    ligands = mol.get_specific_ligand(ligand_type)
    feature = init_feature(len(_properties), operation, depth[1]-depth[0])

    for ligand in ligands:
        scope = set(mol.ligand_ind[ligand])
        feature += multiple_RACs_all_atoms(mol, _properties=_properties, scope=scope, depth=depth, operation=operation, style=style, three_d=three_d)
    
    return np.divide(feature, len(ligands))

def RAC_ligand_ligand(mol, ligand1_index: int, ligand2_index: int, _properties: list, depth: tuple, operation='multiply', style='Moreau-Broto', three_d=False) -> np.ndarray:

    """
    Cross-scope RAC descriptors. Namely, in any atom pair that counts, the first atom belongs to ligand1, and the second atom belongs to ligand2.
    """

    scope_1, scope_2 = set(mol.ligand_ind[ligand1_index]), set(mol.ligand_ind[ligand2_index])
    return multiple_RACs_cross_scope(mol=mol, _properties=_properties, scope_1=scope_1, scope_2=scope_2, depth=depth, operation=operation, style=style, three_d=three_d)

# The following section is only about RAC features for CN/NN ligands. 
# full_RAC_CN_NN follows the RAC-155 list. In our case, it's actually RAC-156.
# updated_RAC_CN_NN includes a few definition changes and more property options. Experimental.

def RAC_graph(names: list) -> simple_graph:

    """
    Grouping the RAC features.
    """

    translation = {
        '0': 'proximal',
        '1': 'proximal',
        '2': 'middle',
        '3': 'distal',
        '3plus': 'distal'
    }

    graph = simple_graph(names)
    graph.get_graph(translation=translation)

    return graph

def RAC_names_CN_NN(depth=(0,3)) -> list:

    """
    Get all the feature names.
    """

    _properties = ['electronegativity', 'atomic number', 'identity', 'covalent radius', 'topology']

    names = []

    def helper(start, scope, operation):
        _new = []
        for _property in _properties:
            if operation == 'multiply' or operation == 'add':
                _new += [feature_name(start, scope, _property, operation, d) for d in range(depth[0], depth[1]+1)]
            else:
                if _property != 'identity':
                    _new += [feature_name(start, scope, _property, operation, d) for d in range(max(1, depth[0]), depth[1]+1)]
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

def full_RAC_CN_NN(mol, depth=(0,3)) -> np.ndarray:

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

def updated_RAC_names_CN_NN() -> list:

    """
    Get all the feature names.
    """

    _properties = ['electronegativity', 'atomic number', 'identity', 'covalent radius', 'topology', 'polarizability', 'vdW radius']

    names = []

    def translate(d):
        if d==3:
            return "3plus"
        else:
            return str(d)

    def helper(start, scope, operation):
        _new = []
        for _property in _properties:
            if operation == 'multiply' or operation == 'add':
                _new += [feature_name(start, scope, _property, operation, translate(d)) for d in range(4)]
            else:
                if _property != 'identity':
                    _new += [feature_name(start, scope, _property, operation, translate(d)) for d in range(1, 4)]
        return _new

    names += helper('f', 'all', 'multiply')
    names += helper('mc', 'CN', 'multiply')
    names += helper('mc', 'NN', 'multiply')
    names += helper('lc', 'CN', 'multiply')
    names += helper('lc', 'NN', 'multiply')
    names += helper('f', 'CN', 'multiply')
    names += helper('f', 'NN', 'multiply')

    names += helper('mc', 'CN', 'subtract')
    names += helper('mc', 'NN', 'subtract')
    names += helper('lc', 'CN', 'subtract')
    names += helper('lc', 'NN', 'subtract')

    return names

def updated_RAC_CN_NN(mol, depth=(0,3)) -> np.ndarray:

    """
    An updated version. Big difference: 3plus stands for depth=3 and greater. Call mol.distance_cheat(fake_depth=3) first. 
    Another big difference: replaced mc/all with mc/CN and mc/NN.
    Experimental, so more properties are included.
    """

    _properties = ['electronegativity', 'atomic number', 'identity', 'covalent radius', 'topology', 'polarizability', 'vdW radius']
    __properties = ['electronegativity', 'atomic number', 'covalent radius', 'topology', 'polarizability', 'vdW radius']

    _f_all = RAC_f_all(mol=mol, _properties=_properties, operation='multiply', depth=depth)
    _mc_CN = RAC_mc_ligand(mol=mol, ligand_type='CN', _properties=_properties, operation='multiply', depth=depth)
    _mc_NN = RAC_mc_ligand(mol=mol, ligand_type='NN', _properties=_properties, operation='multiply', depth=depth)
    _lc_CN = RAC_lc_ligand(mol=mol, ligand_type='CN', _properties=_properties, operation='multiply', depth=depth, average_lc=True)
    _lc_NN = RAC_lc_ligand(mol=mol, ligand_type='NN', _properties=_properties, operation='multiply', depth=depth, average_lc=True)
    _f_CN = RAC_f_ligand(mol=mol, ligand_type='CN', _properties=_properties, operation='multiply', depth=depth)
    _f_NN = RAC_f_ligand(mol=mol, ligand_type='NN', _properties=_properties, operation='multiply', depth=depth)

    __mc_CN = RAC_mc_ligand(mol=mol, ligand_type='CN', _properties=__properties, operation='subtract', depth=depth)
    __mc_NN = RAC_mc_ligand(mol=mol, ligand_type='NN', _properties=__properties, operation='subtract', depth=depth)
    __lc_CN = RAC_lc_ligand(mol=mol, ligand_type='CN', _properties=__properties, operation='subtract', depth=depth, average_lc=True)
    __lc_NN = RAC_lc_ligand(mol=mol, ligand_type='NN', _properties=__properties, operation='subtract', depth=depth, average_lc=True)

    full_feature = np.concatenate((_f_all, _mc_CN, _mc_NN, _lc_CN, _lc_NN, _f_CN, _f_NN, __mc_CN, __mc_NN, __lc_CN, __lc_NN))

    return full_feature