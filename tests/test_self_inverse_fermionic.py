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
from tests._utils import (_validate_block_encoding,
                        _validate_block_encoding_select_is_self_inverse, 
                        _validate_block_encoding_does_nothing_when_control_is_off,
                        _validate_clean_ancillae_are_cleaned,
                        _setup
                        )


MAX_MODES = 7
MAX_ACTIVE_MODES = 7
MIN_ACTIVE_MODES = 1


@pytest.mark.parametrize("trial", range(100))
def test_arbitrary_fermionic_operator_with_hc(trial):
    number_of_active_modes = np.random.randint(MIN_ACTIVE_MODES, MAX_ACTIVE_MODES + 1)
    active_modes = np.random.choice(
        range(MAX_MODES + 1), size=number_of_active_modes, replace=False
    )
    operator_types_reversed = np.random.choice(
        [2, 1, 0], size=number_of_active_modes, replace=True
    )
    while np.allclose(operator_types_reversed, [2] * number_of_active_modes):
        operator_types_reversed = np.random.choice(
            [2, 1, 0], size=number_of_active_modes, replace=True
        )
    operator_types_reversed = operator_types_reversed[:number_of_active_modes]
    operator_types_reversed = list(operator_types_reversed)
    sign = np.random.choice([1, -1])

    operator_string = ""
    for mode, operator_type in zip(active_modes, operator_types_reversed):
        if operator_type == 0:
            operator_string += f" b{mode}"
        if operator_type == 1:
            operator_string += f" b{mode}^"
        if operator_type == 2:
            operator_string += f" b{mode}^ b{mode}"

    operator = ParticleOperator(operator_string, coeff=sign)
    operator += operator.dagger()

    
    expected_rescaling_factor = 1
    maximum_occupation_number = 1

    n_index_qubits = 1
    n_clean_ancilla = 2 
    number_of_block_encoding_ancillae = 1 #Fixed for fermionic BE's
    n_system_qubits = operator.max_fermionic_mode + 1
    n_qubits = n_index_qubits + n_clean_ancilla + number_of_block_encoding_ancillae + n_system_qubits

    qubits = [cirq.LineQubit(i) for i in range(n_qubits)]
    index_register = qubits[:n_index_qubits]
    clean_ancilla_register = qubits[n_index_qubits: n_index_qubits + n_clean_ancilla]
    block_encoding_ancilla_register = qubits[n_index_qubits + n_clean_ancilla:n_index_qubits + n_clean_ancilla + number_of_block_encoding_ancillae]
    system_register = System(1, 
                        len(index_register) + len(clean_ancilla_register) + len(block_encoding_ancilla_register), 
                        operator.max_mode + 1, 0)

    circuit, metrics, system = _setup(
        1,
        operator,
        1,
        partial(
            fermionic_plus_hc_block_encoding,
            active_indices=active_modes[::-1],
            operator_types=operator_types_reversed[::-1],
            sign=sign,
        ),
    )
    

    _validate_clean_ancillae_are_cleaned(
        circuit, system_register, number_of_block_encoding_ancillae
    )
    _validate_block_encoding_does_nothing_when_control_is_off(
        circuit, system_register, number_of_block_encoding_ancillae
    )
    _validate_block_encoding(
        circuit,
        system_register,
        expected_rescaling_factor,
        operator,
        number_of_block_encoding_ancillae,
        maximum_occupation_number,
    )
    _validate_block_encoding_select_is_self_inverse(circuit)


    

def test_fermionic_number_operator_block_encoding_satisfies_qubitization_conditions():
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

    circuit = cirq.Circuit([PREPARE, SELECT, PREPARE_DAGGER])

    _validate_block_encoding(circuit, system_register, rescaling_factor, hamiltonian_operator,len(block_encoding_ancilla_register), 1)
    
    _validate_block_encoding_select_is_self_inverse(circuit)

    _validate_block_encoding_does_nothing_when_control_is_off(circuit, system_register, len(block_encoding_ancilla_register))

    _validate_clean_ancillae_are_cleaned(circuit, system_register, len(block_encoding_ancilla_register))