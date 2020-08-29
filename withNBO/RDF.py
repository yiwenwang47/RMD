from ..elements import property_notation
from ..RDF import RDF_f_all, RDF_f_ligand, RDF_mc_all, RDF_mc_ligand, RDF_ligand_ligand
import numpy as np

def NAO_NPA_names(definitions, betas, distances, option='Singlet') -> list:

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

    def helper(definition):
        _new = []
        for beta in betas:
            for i  in range(len(_notations)):
                _new += ['_'.join([definition[0], definition[1], 'beta'+str(beta), 'd'+str(d), _notations[i]]) for d in distances]
        return _new
    names = []
    for definition in definitions:
        names += helper(definition)

    return names

def NBO_names(definitions, betas, distances, option='singlet') -> list:

    _notations = ['BD_sg_occ', 'BD_sg_nrg', 'BD_db_occ', 'BD_db_nrg', 'ABB_sg_occ', 'ABB_sg_nrg', 'ABB_db_occ', 'ABB_db_nrg']
    
    _suffix = {
        'singlet': '_S',
        'triplet alpha': '_T_a',
        'triplet beta': '_T_b',
        'diff': '_diff'
    }
    _notations = [_notaion+_suffix[option] for _notaion in _notations]

    def helper(definition):
        _new = []
        for beta in betas:
            for i  in range(len(_notations)):
                _new += ['_'.join([definition[0], definition[1], 'beta'+str(beta), 'd'+str(d), _notations[i]]) for d in distances]
        return _new
    names = []
    for definition in definitions:
        names += helper(definition)

    return names


def RDF_full(mol, _properties, distance_range=(1,10), step_size=0.25):

    f_all_b5 = RDF_f_all(mol, _properties=_properties, beta=5, distance_range=distance_range, step_size=step_size)
    f_all_b10 = RDF_f_all(mol, _properties=_properties, beta=10, distance_range=distance_range, step_size=step_size)
    f_all_b20 = RDF_f_all(mol, _properties=_properties, beta=20, distance_range=distance_range, step_size=step_size)
    
    f_CN_b5 = RDF_f_ligand(mol, 'CN', _properties=_properties, beta=5, distance_range=distance_range, step_size=step_size)
    f_CN_b10 = RDF_f_ligand(mol, 'CN', _properties=_properties, beta=10, distance_range=distance_range, step_size=step_size)
    f_CN_b20 = RDF_f_ligand(mol, 'CN', _properties=_properties, beta=20, distance_range=distance_range, step_size=step_size)
    
    
    f_NN_b5 = RDF_f_ligand(mol, 'NN', _properties=_properties, beta=5, distance_range=distance_range, step_size=step_size)
    f_NN_b10 = RDF_f_ligand(mol, 'NN', _properties=_properties, beta=10, distance_range=distance_range, step_size=step_size)
    f_NN_b20 = RDF_f_ligand(mol, 'NN', _properties=_properties, beta=20, distance_range=distance_range, step_size=step_size)

    full_feature = np.concatenate((f_all_b5, f_all_b10, f_all_b20, f_CN_b5, f_CN_b10, f_CN_b20, f_NN_b5, f_NN_b10, f_NN_b20))

    return full_feature

def RDF_mc(mol, _properties, distance_range=(1,10), step_size=0.25):

    mc_all_b5 = RDF_mc_all(mol, _properties=_properties, beta=5, distance_range=distance_range, step_size=step_size)
    mc_all_b10 = RDF_mc_all(mol, _properties=_properties, beta=10, distance_range=distance_range, step_size=step_size)
    mc_all_b20 = RDF_mc_all(mol, _properties=_properties, beta=20, distance_range=distance_range, step_size=step_size)

    mc_CN_b5 = RDF_mc_ligand(mol, 'CN', _properties=_properties, beta=5, distance_range=distance_range, step_size=step_size)
    mc_CN_b10 = RDF_mc_ligand(mol, 'CN', _properties=_properties, beta=10, distance_range=distance_range, step_size=step_size)
    mc_CN_b20 = RDF_mc_ligand(mol, 'CN', _properties=_properties, beta=20, distance_range=distance_range, step_size=step_size)

    mc_NN_b5 = RDF_mc_ligand(mol, 'NN', _properties=_properties, beta=5, distance_range=distance_range, step_size=step_size)
    mc_NN_b10 = RDF_mc_ligand(mol, 'NN', _properties=_properties, beta=10, distance_range=distance_range, step_size=step_size)
    mc_NN_b20 = RDF_mc_ligand(mol, 'NN', _properties=_properties, beta=20, distance_range=distance_range, step_size=step_size)

    full_feature = np.concatenate((mc_all_b5, mc_all_b10, mc_all_b20, mc_CN_b5, mc_CN_b10, mc_CN_b20, mc_NN_b5, mc_NN_b10, mc_NN_b20))

    return full_feature

def RDF_cross_scope(mol, _properties, distance_range=(1,10), step_size=0.25):

    CN1, CN2, NN = mol.CN1, mol.CN2, 3-mol.CN1-mol.CN2

    CN1_NN_b5 = RDF_ligand_ligand(mol, CN1, NN, _properties=_properties, beta=5, distance_range=distance_range, step_size=step_size)
    CN1_NN_b10 = RDF_ligand_ligand(mol, CN1, NN, _properties=_properties, beta=10, distance_range=distance_range, step_size=step_size)
    CN1_NN_b20 = RDF_ligand_ligand(mol, CN1, NN, _properties=_properties, beta=20, distance_range=distance_range, step_size=step_size)

    CN2_NN_b5 = RDF_ligand_ligand(mol, CN2, NN, _properties=_properties, beta=5, distance_range=distance_range, step_size=step_size)
    CN2_NN_b10 = RDF_ligand_ligand(mol, CN2, NN, _properties=_properties, beta=10, distance_range=distance_range, step_size=step_size)
    CN2_NN_b20 = RDF_ligand_ligand(mol, CN2, NN, _properties=_properties, beta=20, distance_range=distance_range, step_size=step_size)

    CN1_CN2_b5 = RDF_ligand_ligand(mol, CN1, CN2, _properties=_properties, beta=5, distance_range=distance_range, step_size=step_size)
    CN1_CN2_b10 = RDF_ligand_ligand(mol, CN1, CN2, _properties=_properties, beta=10, distance_range=distance_range, step_size=step_size)
    CN1_CN2_b20 = RDF_ligand_ligand(mol, CN1, CN2, _properties=_properties, beta=20, distance_range=distance_range, step_size=step_size)

    full_feature = np.concatenate((CN1_NN_b5, CN1_NN_b10, CN1_NN_b20, CN2_NN_b5, CN2_NN_b10, CN2_NN_b20, CN1_CN2_b5, CN1_CN2_b10, CN1_CN2_b20))

    return full_feature