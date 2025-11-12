from src.lobe._utils import (
    get_bosonic_exponents,
    translate_antifermions_to_fermions,
    predict_number_of_block_encoding_ancillae,
)
import numpy as np
from copy import deepcopy
from src.lobe.yukawa import yukawa_term_block_encoding
from src.lobe.system import System
from src.lobe._utils import get_fermionic_operator_types, get_bosonic_exponents
from src.lobe.fermionic import fermionic_product_block_encoding
from src.lobe.bosonic import bosonic_product_block_encoding

CLIFFORD_ANGLES = [0, np.pi/4, np.pi/2, 3 * np.pi/4, np.pi]

def remove_clifford_rotations(angles_list, tol: float = 1e-3):
    non_clifford_angles = []

    for angle in angles_list:
        angle_is_not_clifford = True #assume at first the angle is clifford
        for clifford_angle in CLIFFORD_ANGLES: 
            if np.allclose(angle, clifford_angle, rtol = tol): #For each clifford angle, check if the rotation angle is close to clifford within tolerance
                angle_is_not_clifford = False 
        if angle_is_not_clifford:
            non_clifford_angles.append(angle)


    return np.array(non_clifford_angles)

def get_unique_modes(operator):
    '''
    Given a ParticleOperator, this function returns a list of unique fermionic modes and a list of unique bosonic modes
    '''

    fermionic_modes = []
    bosonic_modes = []

    for term in operator.op_dict.keys():
        for op in term:
            mode = op[1]
            if op[0] == 0:
                fermionic_modes.append(mode)
            elif op[0] == 2:
                bosonic_modes.append(mode)
            else:
                raise Exception( "This function assumes all antifermionic operators are mapped to fermionic ones" )

    return list(set(fermionic_modes)), list(set(bosonic_modes))


def count_metrics(operator, max_occupancy: int = 1):
    metrics = {
        'n_be_ancillae': 0,
        'n_clean_ancillae': 0,
        'n_T_gates': 0,
        'n_non_clifford_rotations': 0,
        'rescaling_factor': 0
    }

    #TODO: map antifermions to fermions

    grouped_operator = operator.group()
    L = len(grouped_operator)
    
    metrics['n_be_ancillae'] = np.ceil(np.log2(L)) + 1 * operator.has_fermions + len(get_unique_modes(operator)[1]) 
    
    for term in grouped_operator:
        if len(term) == 1:
            if term.has_fermions:
                n_clean_ancillae = 1
                rescaling_factor = 1
                n_t_gates = 4
                n_non_clifford_rotations = 0
            elif term.has_bosons:
                n_clean_ancillae = np.ceil(np.log2(max_occupancy))
                rescaling_factor = max_occupancy
                n_t_gates = 7 * np.ceil(np.log2(max_occupancy + 1))

                rotation_angles = np.array([2 * np.arccos(np.sqrt(omega * (omega - 1))/max_occupancy)] for omega in range(0, max_occupancy))
                n_non_clifford_rotations = len(remove_clifford_rotations(rotation_angles))

    metrics['n_T_gates'] = n_t_gates + metrics.get('n_T_gates')
    
    return metrics