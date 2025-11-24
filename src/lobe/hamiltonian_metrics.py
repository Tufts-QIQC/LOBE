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
from src.lobe.metrics import CircuitMetrics

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

def get_rotation_angles(exponents, max_occupancy):
    rotation_angles = []
    for R_and_S in exponents:
        Ri, Si = R_and_S[0], R_and_S[1]
        argument = 1
        for omega in range(Si, max_occupancy - Ri + 1):
            for r in range(0, Ri):
                argument *= np.sqrt(omega - r) / np.sqrt(max_occupancy)
            for s in range(1, Si + 1):
                argument *= np.sqrt(omega - Ri + s ) / np.sqrt(max_occupancy)
            angle = 2 * np.arccos(argument)
            rotation_angles.append(angle)

    return np.array(rotation_angles)

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
    
    B = 0
    
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

                rotation_angles = np.array([2 * np.arccos(np.sqrt(omega * (omega - 1))/max_occupancy) for omega in range(0, max_occupancy)])
                n_non_clifford_rotations = len(remove_clifford_rotations(rotation_angles))
        elif len(term) > 1:
            #assume form of two fermionic ops and 1 or 2 bosonic ops plus h.c.
            _, term_B = get_unique_modes(term[0]) # B = number of unique bosonic modes
            n_boson_ops = term[0].n_bosons

            # Determine rotations
            _, exponents = get_bosonic_exponents(term[0], len(term_B))
            rotation_angles = get_rotation_angles(exponents)
            n_non_clifford_rotations = len(remove_clifford_rotations(rotation_angles))

            if term[0].has_fermions:
                rescaling_factor = max_occupancy ** (n_boson_ops / 2)
                if n_boson_ops == 1:
                    n_clean_ancillae = np.ceil(np.log2(max_occupancy + 1)) + 1
                    n_t_gates = 12 * np.ceil(np.log2(max_occupancy))
                else:
                    n_clean_ancillae = np.ceil(np.log2(max_occupancy)) + 1
                    n_t_gates = 24 * np.ceil(np.log2(max_occupancy)) - 8
                
            else:
                #assume form of n bosonic ops + h.c.
                term_W = np.ceil(np.log2(max_occupancy + 1))
                rescaling_factor = 2 * (max_occupancy ** (n_boson_ops / 2))
                n_clean_ancillae = np.ceil(np.log2(max_occupancy)) + 1
                n_t_gates = 12 * term_B * term_W - 8 * term_B + 4

            B = max(term_B, B)      
        metrics['n_T_gates'] = n_t_gates + metrics.get('n_T_gates')
        metrics['rescaling_factor'] = rescaling_factor + metrics.get('rescaling_factor')
        metrics['n_non_clifford_rotations'] = n_non_clifford_rotations + metrics.get('n_non_clifford_rotations')
        metrics['n_clean_ancillae'] = max(n_clean_ancillae, metrics.get('n_clean_ancillae'))

    #Metrics for indexing over terms
    L = len(grouped_operator)
    metrics['n_T_gates'] = 4 * (L - 1) + metrics.get('n_T_gates')

    metrics['n_be_ancillae'] = np.ceil(np.log2(L)) + 1 * operator.has_fermions + B

    # circuit_metrics = CircuitMetrics()
    # circuit_metrics.number_of_t_gates = metrics['n_T_gates']
    # circuit_metrics.number_of_nonclifford_rotations = metrics['n_non_clifford_rotations']
    
    
    return metrics