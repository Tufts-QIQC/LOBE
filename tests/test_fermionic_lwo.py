import cirq
import pytest
import numpy as np
from functools import partial
from openparticle import ParticleOperator, generate_matrix

import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath('__file__')), '../'))
from src.lobe.fermionic import fermionic_product_block_encoding, fermionic_plus_hc_block_encoding
from src.lobe.system import System
from src.lobe.metrics import CircuitMetrics
from src.lobe.multiplexed_rotations import get_decomposed_multiplexed_rotation_circuit
from src.lobe._utils import _apply_negative_identity, get_basis_of_full_system, get_fermionic_operator_types, pretty_print
from src.lobe.bosonic import _get_bosonic_rotation_angles, _add_multi_bosonic_rotations
from src.lobe.addition import add_classical_value
from src.lobe.decompose import decompose_controls_left, decompose_controls_right
from src.lobe.index import index_over_terms
from src.lobe.asp import add_prepare_circuit, get_target_state
from src.lobe.reflection import add_ancilla_reflection
from src.lobe.qpe_utils import *

def _verify_walker_condition_one(H_matrix, basis, expected_rescaling_factor, PREPARE, SELECT, PREPARE_DAGGER):
    """
    Checks if <0|PREP^\DAGGER SEL PREP|0> = H/ALPHA
    """
    assert np.allclose(
        H_matrix,
        (cirq.Circuit(
            PREPARE,
            SELECT,
            PREPARE_DAGGER 
        ).unitary()*expected_rescaling_factor)[:len(basis),
             :len(basis)]
    ) 

def _verify_walker_condition_two(basis, PREPARE, SELECT, PREPARE_DAGGER):
    """
    Checks if <0|PREP^\DAGGER SEL^2 PREP|0> = 1
    """
    assert np.allclose(
        np.eye(len(basis)),
        (cirq.Circuit(
            PREPARE,
            SELECT,
            SELECT,
            PREPARE_DAGGER
        ).unitary())[:len(basis),
             :len(basis)]
    )





def test_fermionic_number_operator_block_encoding_satisfies_walker_conditions():
    hamiltonian_operator = ParticleOperator('b0^ b0')

    coefficient_vector = [term.coeff if len(term) == 1 else term.coeffs[0] for term in hamiltonian_operator.group()]
    rescaling_factor = np.linalg.norm(coefficient_vector, 1)
    basis = get_basis_of_full_system(1, hamiltonian_operator.max_fermionic_mode + 1, 0)
    H_matrix = generate_matrix(hamiltonian_operator, basis)

    n_index_qubits = 1
    n_clean_ancilla = 2 
    n_be_ancilla = 1 #Fixed for fermionic BE's
    n_system_qubits = hamiltonian_operator.max_fermionic_mode + 1
    n_qubits = n_index_qubits + n_clean_ancilla + n_be_ancilla + n_system_qubits

    qubits = [cirq.LineQubit(i) for i in range(n_qubits)]
    index_register = qubits[:n_index_qubits]
    clean_ancilla_register = qubits[n_index_qubits: n_index_qubits + n_clean_ancilla]
    block_encoding_ancilla_register = qubits[n_index_qubits + n_clean_ancilla:n_index_qubits + n_clean_ancilla + n_be_ancilla]
    system_register = System(1, 
                        len(index_register) + len(clean_ancilla_register) + len(block_encoding_ancilla_register), 
                        hamiltonian_operator.max_mode + 1, 0)


    

    PREPARE = [cirq.X.on(index_register[0])]
    PREPARE_DAGGER = [cirq.X.on(index_register[0])]

    SELECT = fermionic_product_block_encoding(
                    system=system_register,
                    block_encoding_ancillae=block_encoding_ancilla_register,
                    active_indices=[0],
                    operator_types=[2],
                    sign = 1,
                    clean_ancillae=clean_ancilla_register,
                    ctrls = (index_register, [1]),
                )[0]

    

    _verify_walker_condition_one(H_matrix, basis, rescaling_factor, PREPARE, SELECT, PREPARE_DAGGER)
    
    _verify_walker_condition_two(basis, PREPARE, SELECT, PREPARE_DAGGER)

@pytest.mark.parametrize('hamiltonian_operator', 
                         [
                            ParticleOperator('b0^ b1') + ParticleOperator('b1^ b0')
                         ]
                         )
def test_fermionic_product_plus_hc_block_encoding_satisfies_walker_conditions(hamiltonian_operator):

    coefficient_vector = [term.coeff if len(term) == 1 else term.coeffs[0] for term in hamiltonian_operator.group()]
    rescaling_factor = np.linalg.norm(coefficient_vector, 1)
    basis = get_basis_of_full_system(1, hamiltonian_operator.max_fermionic_mode + 1, 0)
    H_matrix = generate_matrix(hamiltonian_operator, basis)

    n_index_qubits = 1
    n_clean_ancilla = 2 
    n_be_ancilla = 1 #Fixed for fermionic BE's
    n_system_qubits = hamiltonian_operator.max_fermionic_mode + 1
    n_qubits = n_index_qubits + n_clean_ancilla + n_be_ancilla + n_system_qubits

    qubits = [cirq.LineQubit(i) for i in range(n_qubits)]
    index_register = qubits[:n_index_qubits]
    clean_ancilla_register = qubits[n_index_qubits: n_index_qubits + n_clean_ancilla]
    block_encoding_ancilla_register = qubits[n_index_qubits + n_clean_ancilla:n_index_qubits + n_clean_ancilla + n_be_ancilla]
    system_register = System(1, 
                        len(index_register) + len(clean_ancilla_register) + len(block_encoding_ancilla_register), 
                        hamiltonian_operator.max_mode + 1, 0)

    active_modes, operator_types = get_fermionic_operator_types(hamiltonian_operator.group()[0].to_list()[0])

    PREPARE = [cirq.X.on(index_register[0])]
    PREPARE_DAGGER = [cirq.X.on(index_register[0])]

    SELECT = fermionic_plus_hc_block_encoding(
                    system=system_register,
                    block_encoding_ancillae=block_encoding_ancilla_register,
                    active_indices=active_modes,
                    operator_types=operator_types,
                    sign = -1,
                    clean_ancillae=clean_ancilla_register,
                    ctrls = (index_register, [1]),
                )[0]
    

    _verify_walker_condition_one(H_matrix, basis, rescaling_factor, PREPARE, SELECT, PREPARE_DAGGER)
    
    _verify_walker_condition_two(basis, PREPARE, SELECT, PREPARE_DAGGER)