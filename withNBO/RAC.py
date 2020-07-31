# First, NAO and NPA(omitted in notation for now). Please see below for detailed explanation on property names.
from ..autocorrelation import *

def NAO_NPA_names(option='Singlet') -> list:

    _properties = ['Weighted energy', 'Natural charge', 
    'Valence s occupancy', 'Valence s energy', 'Valence px occupancy', 'Valence px energy',
    'Valence py occupancy', 'Valence py energy', 'Valence pz occupancy', 'Valence pz energy']
    if option == 'Triplet':
        _properties = _properties + ['Natural Spin Density']
    _notations = [property_notation[_property] for _property in _properties]
    _suffix = {
        'Singlet': '_S',
        'Triplet': '_T',
        'Difference': '_diff'
    }
    _notations = [_notaion+_suffix[option] for _notaion in _notations]

    styles = ['MB', 'M', 'G']

    def helper(start, scope):
        _new = []
        for style in styles:
            for i  in range(len(_properties)):
                _new += ['_'.join([start, scope, style, _notations[i], str(d)]) for d in range(1, 11)]
        return _new

    return helper('f', 'all') + helper('f', 'CN') + helper('f', 'NN')

def RAC_with_NAO_CN_NN(mol, depth=(1,5), option='Singlet', three_d=False) -> np.ndarray:

    _properties = ['Weighted energy', 'Natural charge', 
    'Valence s occupancy', 'Valence s energy', 'Valence px occupancy', 'Valence px energy',
    'Valence py occupancy', 'Valence py energy', 'Valence pz occupancy', 'Valence pz energy']
    _properties = [_property+' '+option for _property in _properties]
    if option == 'Triplet':
        _properties = _properties + ['Natural Spin Density']

    f_all_MB = RAC_f_all(mol, _properties=_properties, depth=depth, operation='multiply', style='Moreau-Broto', three_d=three_d)
    f_all_M = RAC_f_all(mol, _properties=_properties, depth=depth, style='Moran', three_d=three_d)
    f_all_G = RAC_f_all(mol, _properties=_properties, depth=depth, style='Geary', three_d=three_d)
    
    f_CN_MB = RAC_f_ligand(mol, 'CN', _properties=_properties, depth=depth, operation='multiply', style='Moreau-Broto', three_d=three_d)
    f_CN_M = RAC_f_ligand(mol, 'CN', _properties=_properties, depth=depth, style='Moran', three_d=three_d)
    f_CN_G = RAC_f_ligand(mol, 'CN', _properties=_properties, depth=depth, style='Geary', three_d=three_d)
    
    f_NN_MB = RAC_f_ligand(mol, 'NN', _properties=_properties, depth=depth, operation='multiply', style='Moreau-Broto', three_d=three_d)
    f_NN_M = RAC_f_ligand(mol, 'NN', _properties=_properties, depth=depth, style='Moran', three_d=three_d)
    f_NN_G = RAC_f_ligand(mol, 'NN', _properties=_properties, depth=depth, style='Geary', three_d=three_d)
    
    full_feature = np.concatenate((f_all_MB, f_all_M, f_all_G, f_CN_MB, f_CN_M, f_CN_G, f_NN_MB, f_NN_M, f_NN_G))

    return full_feature

def RAC_cross_scope_with_NAO_CN_NN(mol, depth=(1,10), option='Singlet', three_d=False) -> np.ndarray:

    _properties = ['Weighted energy', 'Natural charge', 
    'Valence s occupancy', 'Valence s energy', 'Valence px occupancy', 'Valence px energy',
    'Valence py occupancy', 'Valence py energy', 'Valence pz occupancy', 'Valence pz energy']
    _properties = [_property+' '+option for _property in _properties]
    if option == 'Triplet':
        _properties = _properties + ['Natural Spin Density']

    CN1, CN2, NN = mol.CN1, mol.CN2, mol.get_specific_ligand('NN')[0]

    CN1_NN_MB = RAC_ligand_ligand(mol, CN1, NN, _properties=_properties, depth=depth, operation='multiply', style='Moreau-Broto', three_d=three_d)
    CN1_NN_M = RAC_ligand_ligand(mol, CN1, NN, _properties=_properties, depth=depth, style='Moran', three_d=three_d)
    CN1_NN_G = RAC_ligand_ligand(mol, CN1, NN, _properties=_properties, depth=depth, style='Geary', three_d=three_d)

    CN2_NN_MB = RAC_ligand_ligand(mol, CN2, NN, _properties=_properties, depth=depth, operation='multiply', style='Moreau-Broto', three_d=three_d)
    CN2_NN_M = RAC_ligand_ligand(mol, CN2, NN, _properties=_properties, depth=depth, style='Moran', three_d=three_d)
    CN2_NN_G = RAC_ligand_ligand(mol, CN2, NN, _properties=_properties, depth=depth, style='Geary', three_d=three_d)

    CN1_CN2_MB = RAC_ligand_ligand(mol, CN1, CN2, _properties=_properties, depth=depth, operation='multiply', style='Moreau-Broto', three_d=three_d)
    CN1_CN2_M = RAC_ligand_ligand(mol, CN1, CN2, _properties=_properties, depth=depth, style='Moran', three_d=three_d)
    CN1_CN2_G = RAC_ligand_ligand(mol, CN1, CN2, _properties=_properties, depth=depth, style='Geary', three_d=three_d)

    full_feature = np.concatenate((CN1_NN_MB, CN1_NN_M, CN1_NN_G, CN2_NN_MB, CN2_NN_M, CN2_NN_G, CN1_CN2_MB, CN1_CN2_M, CN1_CN2_G))

    return full_feature