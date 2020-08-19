import numpy as np
# from .elements import *

def radial_distribution_function(array_1: np.ndarray, d_matrix: np.ndarray, array_2: np.ndarray, \
    beta: float, R: float, with_origin=False, cross_scope=False) -> np.float:

    """
    A very simple radial distribution function. If starts from one atom, set with_origin=True.
    """

    assert not (with_origin and cross_scope)
    matrix = d_matrix - R
    matrix = np.exp(matrix * matrix *(-beta))
    matrix[np.where(d_matrix==0)] = 0
    f_inv = matrix.sum() #inverse of scaling factor
    if f_inv == 0:
        return 0
    else:
        f = 1/f_inv
    if with_origin:
        assert len(matrix.shape) == 1
        return (array_1 * matrix).dot(array_2)  * f
    else:
        if cross_scope:
            assert matrix.shape[0] == len(array_1) and matrix.shape[1] == len(array_2)
        else:
            assert len(matrix.shape) == 2
            assert ((array_1-array_2)**2).sum() < 1e-5
        return array_1.dot(matrix).dot(array_2) * f

def RDFs_from_atom(mol, _properties: list, origin: int, scope: set, beta: float, distance_range: tuple, step_size: float) -> np.ndarray:
    
    """
    Multiple RDFs weighted by a list of properties from an origin atom.
    The parameter scope is forced to be a set just to prevent possible replicates. Pass scope = set() to get a full-scope feature.
    """

    _length = int((distance_range[1] - distance_range[0])/step_size + 1)
    distances = np.linspace(distance_range[0], distance_range[1], _length)
    d_from_origin = mol.matrix[origin].copy()
    if scope:
        scope = list(scope)
        d_from_origin = d_from_origin[scope]

    for i, _property in enumerate(_properties):
        mol.populate_property(_property)
        _new = np.zeros(_length).astype(np.float)
        array_2 = mol.properties[_property].copy()
        if scope:
            array_2 = array_2[scope]
        array_1 = mol.properties[_property][origin] * np.ones(len(d_from_origin))
        for j, R in enumerate(distances):
            _new[j] = radial_distribution_function(array_1, d_from_origin, array_2, beta, R, with_origin=True)
        if i==0:
            feature = _new
        else:
            feature = np.concatenate((feature, _new))

    return feature

def RDFs_all_atoms(mol, _properties: list, scope: set, beta: float, distance_range: tuple, step_size: float) -> np.ndarray:

    """
    Multiple RDFs weighted by a list of properties between all atoms in the defined scope.
    The parameter scope is forced to be a set just to prevent possible replicates. Pass scope = set() to get a full-scope feature.
    """

    
    _length = int((distance_range[1] - distance_range[0])/step_size + 1)
    distances = np.linspace(distance_range[0], distance_range[1], _length)
    matrix = mol.matrix.copy()
    if scope:
        scope = list(scope)
        matrix = matrix[scope][:, scope]

    for i, _property in enumerate(_properties):
        mol.populate_property(_property)
        _new = np.zeros(_length).astype(np.float)
        array = mol.properties[_property].copy()
        if scope:
            array = array[scope]
        for j, R in enumerate(distances):
            _new[j] = radial_distribution_function(array, matrix, array, beta, R)
        if i==0:
            feature = _new
        else:
            feature = np.concatenate((feature, _new))

    return feature

def RDFs_cross_scope(mol, _properties: list, scope_1: set, scope_2:set, beta: float, distance_range: tuple, step_size: float) -> np.ndarray:

    """
    Multiple cross-scope RDFs weighted by a list of properties. 
    For any valid atom pair, the first atom belongs to scope_1, and the second belongs to scope_2.
    """

    assert len(scope_1) > 0 and len(scope_2) > 0
    scope_1, scope_2 = list(scope_1), list(scope_2)
    _length = int((distance_range[1] - distance_range[0])/step_size + 1)
    distances = np.linspace(distance_range[0], distance_range[1], _length)
    matrix = mol.matrix.copy()[scope_1][:, scope_2]
    
    for i, _property in enumerate(_properties):
        mol.populate_property(_property)
        _new = np.zeros(_length).astype(np.float)
        array = mol.properties[_property].copy()
        array_1, array_2 = array[scope_1], array[scope_2]
        for j, R in enumerate(distances):
            _new[j] = radial_distribution_function(array_1, matrix, array_2, beta, R, cross_scope=True)
        if i==0:
            feature = _new
        else:
            feature = np.concatenate((feature, _new))

    return feature

def RDF_f_all(mol, _properties: list, beta: float, distance_range: tuple, step_size: float) -> np.ndarray:
    return RDFs_all_atoms(mol, _properties=_properties, scope=set(), beta=beta, distance_range=distance_range, step_size=step_size)

def RDF_mc_all(mol, _properties: list, beta: float, distance_range: tuple, step_size: float, average_mc=True) -> np.ndarray:

    """
    The feature vector is averaged over all metal centers by default.
    """

    n_mc = len(mol.mcs)
    assert n_mc > 0

    for i, mc in enumerate(mol.mcs):
        _new = RDFs_from_atom(mol, _properties=_properties, origin=mc, scope=set(), beta=beta, distance_range=distance_range, step_size=step_size)
        if i == 0:
            feature = _new
        else:
            feature += _new
    
    if not average_mc:
        return feature
    return np.divide(feature, n_mc)

def RDF_mc_ligand(mol, ligand_type: str, _properties: list, beta: float, distance_range: tuple, step_size: float, average_mc=True) -> np.ndarray:

    """
    The feature vector is averaged over all metal centers and all ligands by default.
    """

    n_mc = len(mol.mcs)
    ligands = mol.get_specific_ligand(ligand_type)
    assert n_mc > 0

    for i, mc in enumerate(mol.mcs):
        for j, ligand in enumerate(ligands):
            scope = set(mol.ligand_ind[ligand])
            scope.update([mc])
            _new = RDFs_from_atom(mol, _properties=_properties, origin=mc, scope=scope, beta=beta, distance_range=distance_range, step_size=step_size)
            if i+j == 0:
                feature = _new
            else:
                feature += _new

