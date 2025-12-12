import cirq
import numpy as np
from openparticle import ParticleOperator
from copy import deepcopy

from .addition import _get_p_val, add_classical_value
from .bosonic import bosonic_product_block_encoding, _get_bosonic_rotation_angles
from .fermionic import fermionic_product_block_encoding
from .interaction import interaction_term_block_encoding
from .metrics import CircuitMetrics
from .multiplexed_rotations import _process_rotation_angles
from .system import System

from ._utils import (
    get_fermionic_operator_types, 
    get_bosonic_exponents, 
    translate_antifermions_to_fermions, 
    get_active_bosonic_modes, 
    get_active_fermionic_modes,
    predict_number_of_block_encoding_ancillae
)

def _compute_n_elbows_and_anc_hw_per_incrementer(N, m):
    control = [cirq.LineQubit(0)]
    clean_ancillae = [cirq.LineQubit(i) for i in range(-1, -100, -1)]   
    """
    |N⟩ -> |N + m⟩
    Returns:
        number of elbows and max clean ancillae to implement this operation
    """

    main_register = [cirq.LineQubit(i) for i in range(1, N + 1)]
    _, metrics = add_classical_value(
        main_register,
        m,
        clean_ancillae=clean_ancillae,
        ctrls = (control, [1])
    )
    return metrics.number_of_elbows, metrics.ancillae_highwater()

def count_metrics(operator, max_occupancy: int = 1):

    groups = operator.group()

    if operator.has_antifermions:
        translated_groups = []
        max_fermionic_mode = operator.max_fermionic_mode
        for term in groups:
            translated_groups.append(
                translate_antifermions_to_fermions(term, max_fermionic_mode)
            )
        groups = translated_groups

    L = len(groups)
    number_of_indexing_clean_ancillae = np.ceil(np.log2(L))

    metrics = CircuitMetrics()

    B = 0
    bosonic_index_qubit = False

    for term in groups:
        B = max(predict_number_of_block_encoding_ancillae(term), B)

        if len(term) == 1:
            active_fermionic_modes = get_active_fermionic_modes(term)
            active_bosonic_modes, exponents_list = get_bosonic_exponents(term)
            P = sum([exponents[0] + exponents[1] for exponents in exponents_list])

            if term.has_fermions and not term.has_bosons: #bi^ bi, bi^ bi bj^ bj, ...
                rescaling_factor = 1
                number_of_elbows = len(active_fermionic_modes)
                clean_ancillae_usage = [len(active_fermionic_modes) + number_of_indexing_clean_ancillae]
                rotation_angles = []
            elif term.has_bosons and not term.has_fermions: #ai^ ai, ai^ ai aj^ aj, ...
                rescaling_factor = max_occupancy  ** (P/2)
                number_of_elbows = int(np.ceil(np.log2(max_occupancy + 1))) * len((active_bosonic_modes))
                clean_ancillae_usage = [
                    i + number_of_indexing_clean_ancillae for i in range(1, 
                                            int(np.ceil(np.log2(max_occupancy + 1))) + 1)
                ]
                rotation_angles = []
                for exponents in exponents_list:
                    angles = _get_bosonic_rotation_angles(maximum_occupation_number=max_occupancy,
                                                                creation_exponent=exponents[0],
                                                                annihilation_exponent=exponents[1])
                    angles = np.concatenate(
                        [angles, np.zeros((1 << int(np.ceil(np.log2(len(angles))))) - len(angles))]
                    )
                    processed_angles = list(_process_rotation_angles(angles))
                    rotation_angles += processed_angles + [-sum(processed_angles)/2, sum(processed_angles)/2]

            else: #bi^ bi ai^ ai
                rescaling_factor = max_occupancy  ** (P/2) 
                clean_ancillae_usage = [len(active_fermionic_modes) + number_of_indexing_clean_ancillae] +\
                        [
                    i + number_of_indexing_clean_ancillae for i in range(1, 
                                            int(np.ceil(np.log2(max_occupancy + 1))) + 1)
                ]
                rotation_angles = []
                for exponents in exponents_list:
                    angles = _get_bosonic_rotation_angles(maximum_occupation_number=max_occupancy,
                                                                creation_exponent=exponents[0],
                                                                annihilation_exponent=exponents[1])
                    angles = np.concatenate(
                        [angles, np.zeros((1 << int(np.ceil(np.log2(len(angles))))) - len(angles))]
                    )
                    processed_angles = list(_process_rotation_angles(angles))
                    rotation_angles += processed_angles + [-sum(processed_angles)/2, sum(processed_angles)/2]

                number_of_elbows = (len(active_fermionic_modes)) +\
                      int(np.ceil(np.log2(max_occupancy + 1))) * len(active_bosonic_modes)
        elif len(term) > 1:
            #term + h.c.
            term = term.to_list()[0]

            active_fermionic_modes = get_active_fermionic_modes(term)
            active_bosonic_modes = get_active_bosonic_modes(term)

            # Determine rotations
            _, exponents_list = get_bosonic_exponents(term, term.max_mode + 1)
            P = sum([exponents_list[i][0] + exponents_list[i][1] for i in range(len(exponents_list))])
            RS_list = [exponents_list[i][0] - exponents_list[i][1] for i in range(len(exponents_list))]
            
            rotation_angles = []
            for exponents in exponents_list:
                angles = _get_bosonic_rotation_angles(maximum_occupation_number=max_occupancy,
                                                            creation_exponent=exponents[0],
                                                            annihilation_exponent=exponents[1])
                angles = np.concatenate(
                    [angles, np.zeros((1 << int(np.ceil(np.log2(len(angles))))) - len(angles))]
                )
                processed_angles = list(_process_rotation_angles(angles))
                rotation_angles += processed_angles + [-sum(processed_angles)/2, sum(processed_angles)/2]

            if term.has_fermions and not term.has_bosons: #e.g. bi^ bj + bj^ bi
                rescaling_factor = 1
                clean_ancillae_usage = [len(active_fermionic_modes) - 1 + number_of_indexing_clean_ancillae]
                number_of_elbows = len(active_fermionic_modes) - 1
            
            elif term.has_bosons and not term.has_fermions: #e.g. ai^ aj + aj^ ai
                rescaling_factor =  (max_occupancy ** (P / 2))
                clean_ancillae_usage = [
                    i + number_of_indexing_clean_ancillae for i in range(1, int(np.ceil(np.log2(max_occupancy + 1))) + 1 + 1)
                ]
                
                number_of_elbows = 1 + len(active_bosonic_modes) * int(np.ceil(np.log2(max_occupancy + 1)))

                for RminusS in RS_list: #+R - S
                    elbows, clean_anc = _compute_n_elbows_and_anc_hw_per_incrementer(
                        int(np.ceil(np.log2(max_occupancy + 1))), RminusS
                    )
                    number_of_elbows += elbows
                    clean_ancillae_usage.append(clean_anc)
                    elbows, clean_anc = _compute_n_elbows_and_anc_hw_per_incrementer(
                        int(np.ceil(np.log2(max_occupancy + 1))), -RminusS
                    )
                    number_of_elbows += elbows
                    clean_ancillae_usage.append(clean_anc)
                    
                    
                # bosonic_index_qubit = True

            else: # e.g. bi^ bj ak + h.c.
                rescaling_factor = max_occupancy ** (P/2)
                clean_ancillae_usage = [
                            i + number_of_indexing_clean_ancillae
                            for i in range(
                                1, int(np.ceil(np.log2(max_occupancy + 1))) + 1 + 1
                            )
                        ]
                number_of_elbows = (len(active_fermionic_modes))  +\
                             len(active_bosonic_modes) * int(np.ceil(np.log2(max_occupancy + 1)))
                for RminusS in RS_list: #+R - S
                    elbows, clean_anc = _compute_n_elbows_and_anc_hw_per_incrementer(
                        int(np.ceil(np.log2(max_occupancy + 1))), RminusS
                    )
                    number_of_elbows += elbows
                    clean_ancillae_usage.append(clean_anc)
                    elbows, clean_anc = _compute_n_elbows_and_anc_hw_per_incrementer(
                        int(np.ceil(np.log2(max_occupancy + 1))), -RminusS
                    )
                    number_of_elbows += elbows
                    clean_ancillae_usage.append(clean_anc)
        

        metrics.number_of_elbows += number_of_elbows
        metrics.clean_ancillae_usage += clean_ancillae_usage
        metrics.rescaling_factor += rescaling_factor
        metrics.rotation_angles += rotation_angles


    metrics.number_of_be_ancillae = np.ceil(np.log2(L)) + B + int(bosonic_index_qubit)
    metrics.number_of_elbows += L - 1 # number of left elbows from indexing 

    return metrics
