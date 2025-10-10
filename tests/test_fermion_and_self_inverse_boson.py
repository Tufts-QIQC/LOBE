import cirq
import pytest
import numpy as np
from functools import partial
from openparticle import ParticleOperator, generate_matrix

import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath('__file__')), '../'))
from src.lobe.self_inverse_boson import self_inverse_bosonic_number_operator_block_encoding, self_inverse_bosonic_product_plus_hc_block_encoding
from src.lobe.system import System
from src.lobe.asp import add_prepare_circuit, get_target_state
from _utils import get_basis_of_full_system
from src.lobe.fermionic import fermionic_product_block_encoding, fermionic_plus_hc_block_encoding
from src.lobe.multiplexed_rotations import get_decomposed_multiplexed_rotation_circuit
from src.lobe._utils import _apply_negative_identity, get_basis_of_full_system, get_fermionic_operator_types, pretty_print
from src.lobe.bosonic import _get_bosonic_rotation_angles, _add_multi_bosonic_rotations
from src.lobe.addition import add_classical_value
from src.lobe.decompose import decompose_controls_left, decompose_controls_right
from src.lobe.index import index_over_terms


def _verify_qubitization_condition_one(H_matrix, basis, expected_rescaling_factor, PREPARE, SELECT, PREPARE_DAGGER):
    """
    Checks if <0|PREP^\DAGGER SEL PREP|0> = H/ALPHA
    """

    unitary = cirq.Circuit(
            PREPARE,
            SELECT,
            PREPARE_DAGGER 
        ).unitary()
    rescaled_block = (unitary * expected_rescaling_factor)[:len(basis), :len(basis)]
    assert np.allclose(
        H_matrix,
        rescaled_block
    ) 

def _verify_qubitization_condition_two(basis, PREPARE, SELECT, PREPARE_DAGGER):
    """
    Checks if <0|PREP^\DAGGER SEL^2 PREP|0> = 1
    """
    unitary = cirq.Circuit(
            PREPARE,
            SELECT,
            SELECT,
            PREPARE_DAGGER 
        ).unitary()
    assert np.allclose(
        np.eye(len(basis)),
        unitary[:len(basis),:len(basis)]
    )



def test_fermionic_bosonic_product_plus_hc_satisfies_qubitization_conditions():
    operator = ParticleOperator('b0^ b1 a0')
    operator += operator.dagger()
    maximum_occupation_number = 3

    active_boson_modes = [0]
    active_fermion_modes = [0, 1]
    boson_exponents_list = [(0, 1)]
    fermion_operator_types = [0, 1]

    coefficient_vector = [term.coeff if len(term) == 1 else term.coeffs[0] for term in operator.group()]
    rescaling_factor = np.linalg.norm(coefficient_vector, 1)
    basis = get_basis_of_full_system(maximum_occupation_number, operator.max_fermionic_mode + 1, operator.max_bosonic_mode + 1)
    H_matrix = generate_matrix(operator, basis)

    n_clean_ancillae = 10
    index_register = [cirq.LineQubit(0)]
    clean_ancilla_register = [cirq.LineQubit(i) for i in range(1, n_clean_ancillae + 1)]
    be_anc_register = [cirq.LineQubit(i) for i in range(n_clean_ancillae + 1, n_clean_ancillae + 1 + 2 + len(active_boson_modes) + 1)]
    system = System(maximum_occupation_number, 
                    len(index_register) + len(clean_ancilla_register) + len(be_anc_register),
                    operator.max_fermionic_mode + 1, 
                    operator.max_bosonic_mode + 1)


    PREPARE = []
    for bosonic_reg in system.bosonic_modes:
        PREPARE.append(cirq.I.on_each(*bosonic_reg))
    PREPARE.append(cirq.X.on(index_register[0]))

    SELECT, _ = self_inverse_bosonic_product_plus_hc_block_encoding(
        system = system,
        block_encoding_ancillae=be_anc_register[:-1],
        active_indices=active_boson_modes,
        exponents_list=boson_exponents_list,
        sign = -1,
        clean_ancillae=clean_ancilla_register,
        ctrls = ([index_register[0]], [1])
    )

    SELECT += fermionic_plus_hc_block_encoding(
                    system=system,
                    block_encoding_ancillae=[be_anc_register[-1]],
                    active_indices=active_fermion_modes,
                    operator_types=fermion_operator_types,
                    sign = -1,
                    clean_ancillae=clean_ancilla_register,
                    ctrls = ([index_register[0]], [1]),
                )[0]

    PREPARE_DAGGER = [cirq.X.on(index_register[0])]


    

    _verify_qubitization_condition_one(H_matrix, basis, rescaling_factor, PREPARE, SELECT, PREPARE_DAGGER)
    
    _verify_qubitization_condition_two(basis, PREPARE, SELECT, PREPARE_DAGGER)