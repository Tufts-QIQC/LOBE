from openparticle import ParticleOperator, generate_matrix
import numpy as np
import cirq
from src.lobe.system import System
from src.lobe._utils import get_basis_of_full_system
from src.lobe.custom import (
    yukawa_3point_pair_term_block_encoding,
    yukawa_4point_pair_term_block_encoding,
)
import pytest
import math


def _validate_block_encoding(
    expected_rescaling_factor,
    operator,
    number_of_block_encoding_ancillae,
    maximum_occupation_number,
    active_indices,
    block_encoding_function,
):
    number_of_modes = max([term.max_mode for term in operator.to_list()]) + 1

    number_of_clean_ancillae = 100

    # Declare Qubits
    circuit = cirq.Circuit()
    control = cirq.LineQubit(0)
    block_encoding_ancillae = [
        cirq.LineQubit(i + 1) for i in range(number_of_block_encoding_ancillae)
    ]

    clean_ancillae = [
        cirq.LineQubit(i + 1 + number_of_block_encoding_ancillae)
        for i in range(number_of_clean_ancillae)
    ]
    system = System(
        number_of_modes=number_of_modes,
        maximum_occupation_number=maximum_occupation_number,
        number_of_used_qubits=1
        + number_of_block_encoding_ancillae
        + number_of_clean_ancillae,
        has_fermions=operator.has_fermions,
        has_bosons=operator.has_bosons,
    )
    circuit.append(
        cirq.I.on_each(
            control,
            *block_encoding_ancillae,
            *system.fermionic_register,
        )
    )
    for bosonic_reg in system.bosonic_system:
        circuit.append(cirq.I.on_each(*bosonic_reg))

    if len(circuit.all_qubits()) >= 32:
        pytest.skip(f"too many qubits to build circuit: {len(circuit.all_qubits())}")

    # Flip control qubit so that we can focus on the 0-subspace of the control
    circuit.append(cirq.X.on(control))
    # Generate full Block-Encoding circuit
    gates, metrics = block_encoding_function(
        system,
        block_encoding_ancillae,
        active_indices,
        clean_ancillae=clean_ancillae,
        ctrls=([control], [1]),
    )
    circuit += gates
    # Flip control qubit so that we can focus on the 0-subspace of the control
    circuit.append(cirq.X.on(control))

    if len(circuit.all_qubits()) >= 20:
        pytest.skip(f"too many qubits to validate: {len(circuit.all_qubits())}")
    elif len(circuit.all_qubits()) >= 15:
        print("Testing singular quantum state")
        simulator = cirq.Simulator()
        random_system_state = 1j * np.random.uniform(
            -1, 1, 1 << system.number_of_system_qubits
        )
        random_system_state += np.random.uniform(
            -1, 1, 1 << system.number_of_system_qubits
        )
        random_system_state = random_system_state / np.linalg.norm(random_system_state)
        zero_state = np.zeros(
            1
            << (
                len(circuit.all_qubits())
                - system.number_of_system_qubits
                - 1
                - number_of_block_encoding_ancillae
            ),
            dtype=complex,
        )
        zero_state[0] = 1
        initial_control_state = [1, 0]
        initial_block_encoding_ancilla_state = np.zeros(
            1 << number_of_block_encoding_ancillae
        )
        initial_block_encoding_ancilla_state[0] = 1
        initial_state = np.kron(
            np.kron(
                np.kron(initial_control_state, initial_block_encoding_ancilla_state),
                zero_state,
            ),
            random_system_state,
        )
        output_state = simulator.simulate(
            circuit, initial_state=initial_state
        ).final_state_vector
        final_state = output_state[: 1 << system.number_of_system_qubits]
        full_fock_basis = get_basis_of_full_system(
            number_of_modes,
            maximum_occupation_number,
            has_fermions=operator.has_fermions,
            has_antifermions=operator.has_antifermions,
            has_bosons=operator.has_bosons,
        )
        matrix = generate_matrix(operator, full_fock_basis)

        expected_final_state = matrix @ random_system_state
        expected_final_state = expected_final_state / np.linalg.norm(
            expected_final_state
        )
        assert np.allclose(
            expected_final_state, final_state / np.linalg.norm(final_state), atol=1e-6
        )
    else:
        full_fock_basis = get_basis_of_full_system(
            number_of_modes,
            maximum_occupation_number,
            has_fermions=operator.has_fermions,
            has_antifermions=operator.has_antifermions,
            has_bosons=operator.has_bosons,
        )
        matrix = generate_matrix(operator, full_fock_basis)

        upper_left_block = circuit.unitary(dtype=complex)[
            : 1 << system.number_of_system_qubits, : 1 << system.number_of_system_qubits
        ]

        assert np.allclose(expected_rescaling_factor * upper_left_block, matrix)


def _validate_clean_ancillae_are_cleaned(
    operator,
    number_of_block_encoding_ancillae,
    maximum_occupation_number,
    active_indices,
    block_encoding_function,
):
    number_of_modes = max([term.max_mode for term in operator.to_list()]) + 1

    number_of_clean_ancillae = 100

    # Declare Qubits
    circuit = cirq.Circuit()
    control = cirq.LineQubit(0)
    block_encoding_ancillae = [
        cirq.LineQubit(i + 1) for i in range(number_of_block_encoding_ancillae)
    ]

    clean_ancillae = [
        cirq.LineQubit(i + 1 + number_of_block_encoding_ancillae)
        for i in range(number_of_clean_ancillae)
    ]
    system = System(
        number_of_modes=number_of_modes,
        maximum_occupation_number=maximum_occupation_number,
        number_of_used_qubits=1
        + number_of_block_encoding_ancillae
        + number_of_clean_ancillae,
        has_fermions=operator.has_fermions,
        has_bosons=operator.has_bosons,
    )
    circuit.append(
        cirq.I.on_each(
            control,
            *block_encoding_ancillae,
            *system.fermionic_register,
        )
    )
    for bosonic_reg in system.bosonic_system:
        circuit.append(cirq.I.on_each(*bosonic_reg))

    if len(circuit.all_qubits()) >= 32:
        pytest.skip(f"too many qubits to build circuit: {len(circuit.all_qubits())}")

    # Flip control qubit so that we can focus on the 0-subspace of the control
    circuit.append(cirq.X.on(control))
    # Generate full Block-Encoding circuit
    gates, metrics = block_encoding_function(
        system,
        block_encoding_ancillae,
        active_indices,
        clean_ancillae=clean_ancillae,
        ctrls=([control], [1]),
    )
    circuit += gates
    # Flip control qubit so that we can focus on the 0-subspace of the control
    circuit.append(cirq.X.on(control))

    if len(circuit.all_qubits()) >= 20:
        pytest.skip(
            f"too many qubits to explicitly validate: {len(circuit.all_qubits())}"
        )
    else:
        simulator = cirq.Simulator()
        random_system_state = 1j * np.random.uniform(
            -1, 1, 1 << system.number_of_system_qubits
        )
        random_system_state += np.random.uniform(
            -1, 1, 1 << system.number_of_system_qubits
        )
        random_system_state = random_system_state / np.linalg.norm(random_system_state)
        zero_state = np.zeros(
            1
            << (
                len(circuit.all_qubits())
                - system.number_of_system_qubits
                - 1
                - number_of_block_encoding_ancillae
            ),
            dtype=complex,
        )
        zero_state[0] = 1
        initial_control_state = 1j * np.random.uniform(-1, 1, 2) + np.random.uniform(
            -1, 1, 2
        )
        initial_control_state = initial_control_state / np.linalg.norm(
            initial_control_state
        )
        initial_block_encoding_ancilla_state = 1j * np.random.uniform(
            -1, 1, 1 << number_of_block_encoding_ancillae
        )
        initial_block_encoding_ancilla_state += np.random.uniform(
            -1, 1, 1 << number_of_block_encoding_ancillae
        )
        initial_block_encoding_ancilla_state = (
            initial_block_encoding_ancilla_state
            / np.linalg.norm(initial_block_encoding_ancilla_state)
        )
        initial_state = np.kron(
            np.kron(
                np.kron(initial_control_state, initial_block_encoding_ancilla_state),
                zero_state,
            ),
            random_system_state,
        )
        final_state = simulator.simulate(
            circuit, initial_state=initial_state
        ).final_state_vector

        indices = final_state.nonzero()[0]
        for index in indices:
            bitstring = format(index, f"0{2+len(circuit.all_qubits())}b")[2:]
            for bit in bitstring[
                1 + number_of_block_encoding_ancillae : -system.number_of_system_qubits
            ]:
                if bit == "1":
                    assert False


def _validate_block_encoding_does_nothing_when_control_is_off(
    operator,
    number_of_block_encoding_ancillae,
    maximum_occupation_number,
    active_indices,
    block_encoding_function,
):
    number_of_modes = max([term.max_mode for term in operator.to_list()]) + 1

    number_of_clean_ancillae = 100

    # Declare Qubits
    circuit = cirq.Circuit()
    control = cirq.LineQubit(0)
    block_encoding_ancillae = [
        cirq.LineQubit(i + 1) for i in range(number_of_block_encoding_ancillae)
    ]

    clean_ancillae = [
        cirq.LineQubit(i + 1 + number_of_block_encoding_ancillae)
        for i in range(number_of_clean_ancillae)
    ]
    system = System(
        number_of_modes=number_of_modes,
        maximum_occupation_number=maximum_occupation_number,
        number_of_used_qubits=1
        + number_of_block_encoding_ancillae
        + number_of_clean_ancillae,
        has_fermions=operator.has_fermions,
        has_bosons=operator.has_bosons,
    )
    circuit.append(
        cirq.I.on_each(
            control,
            *block_encoding_ancillae,
            *system.fermionic_register,
        )
    )
    for bosonic_reg in system.bosonic_system:
        circuit.append(cirq.I.on_each(*bosonic_reg))

    if len(circuit.all_qubits()) >= 32:
        pytest.skip(f"too many qubits to build circuit: {len(circuit.all_qubits())}")

    # Generate full Block-Encoding circuit
    gates, metrics = block_encoding_function(
        system,
        block_encoding_ancillae,
        active_indices,
        clean_ancillae=clean_ancillae,
        ctrls=([control], [1]),
    )
    circuit += gates

    if len(circuit.all_qubits()) >= 20:
        pytest.skip(f"too many qubits to validate: {len(circuit.all_qubits())}")
    else:
        simulator = cirq.Simulator()
        random_system_state = 1j * np.random.uniform(
            -1, 1, 1 << system.number_of_system_qubits
        )
        random_system_state += np.random.uniform(
            -1, 1, 1 << system.number_of_system_qubits
        )
        random_system_state = random_system_state / np.linalg.norm(random_system_state)
        zero_state = np.zeros(
            1
            << (
                len(circuit.all_qubits())
                - system.number_of_system_qubits
                - 1
                - number_of_block_encoding_ancillae
            ),
            dtype=complex,
        )
        zero_state[0] = 1
        initial_control_state = [1, 0]
        initial_block_encoding_ancilla_state = np.zeros(
            1 << number_of_block_encoding_ancillae
        )
        initial_block_encoding_ancilla_state[0] = 1
        initial_state = np.kron(
            np.kron(
                np.kron(initial_control_state, initial_block_encoding_ancilla_state),
                zero_state,
            ),
            random_system_state,
        )
        output_state = simulator.simulate(
            circuit, initial_state=initial_state
        ).final_state_vector
        final_state = output_state[: 1 << system.number_of_system_qubits]

        assert np.allclose(random_system_state, final_state, atol=1e-6)


def _check_metrics_yukawa_3point(
    operator,
    maximum_occupation_number,
    active_indices,
    block_encoding_function,
):
    number_of_modes = max([term.max_mode for term in operator.to_list()]) + 1

    number_of_clean_ancillae = 100

    # Declare Qubits
    circuit = cirq.Circuit()
    control = cirq.LineQubit(0)
    block_encoding_ancillae = [cirq.LineQubit(1), cirq.LineQubit(2)]

    clean_ancillae = [cirq.LineQubit(i + 3) for i in range(number_of_clean_ancillae)]
    system = System(
        number_of_modes=number_of_modes,
        maximum_occupation_number=maximum_occupation_number,
        number_of_used_qubits=3 + number_of_clean_ancillae,
        has_fermions=operator.has_fermions,
        has_bosons=operator.has_bosons,
    )
    circuit.append(
        cirq.I.on_each(
            control,
            *block_encoding_ancillae,
            *system.fermionic_register,
        )
    )
    for bosonic_reg in system.bosonic_system:
        circuit.append(cirq.I.on_each(*bosonic_reg))

    if len(circuit.all_qubits()) >= 32:
        pytest.skip(f"too many qubits to build circuit: {len(circuit.all_qubits())}")

    # Generate full Block-Encoding circuit
    _, metrics = block_encoding_function(
        system,
        block_encoding_ancillae,
        active_indices,
        clean_ancillae=clean_ancillae,
        ctrls=([control], [1]),
    )


MAX_NUMBER_OF_BOSONIC_MODES = 3
MAX_NUMBER_OF_FERMIONIC_MODES = 5
POSSIBLE_MAX_OCCUPATION_NUMBERS = [1, 3, 7]


@pytest.mark.parametrize("trial", range(100))
def test_yukawa_3point(trial):
    maximum_occupation_number = int(
        np.random.choice(POSSIBLE_MAX_OCCUPATION_NUMBERS, size=1)
    )
    number_of_bosonic_modes = np.random.random_integers(1, MAX_NUMBER_OF_BOSONIC_MODES)
    number_of_fermionic_modes = np.random.random_integers(
        2, MAX_NUMBER_OF_FERMIONIC_MODES
    )
    bosonic_index = list(np.random.choice(range(number_of_bosonic_modes), size=1))
    fermionic_indices = list(
        np.random.choice(range(number_of_fermionic_modes), size=2, replace=False)
    )

    operator_string = (
        f"b{fermionic_indices[1]} b{fermionic_indices[0]} a{bosonic_index[0]}^"
    )
    operator = ParticleOperator(operator_string)
    conjugate_string = (
        f"b{fermionic_indices[0]}^ b{fermionic_indices[1]}^ a{bosonic_index[0]}"
    )
    operator += ParticleOperator(conjugate_string)

    _validate_clean_ancillae_are_cleaned(
        operator,
        2,
        maximum_occupation_number,
        bosonic_index + fermionic_indices,
        yukawa_3point_pair_term_block_encoding,
    )
    _validate_block_encoding_does_nothing_when_control_is_off(
        operator,
        2,
        maximum_occupation_number,
        bosonic_index + fermionic_indices,
        yukawa_3point_pair_term_block_encoding,
    )
    _validate_block_encoding(
        np.sqrt(maximum_occupation_number),
        operator,
        2,
        maximum_occupation_number,
        bosonic_index + fermionic_indices,
        yukawa_3point_pair_term_block_encoding,
    )
    _check_metrics_yukawa_3point(
        operator,
        maximum_occupation_number,
        bosonic_index + fermionic_indices,
        yukawa_3point_pair_term_block_encoding,
    )


@pytest.mark.parametrize("trial", range(100))
def test_yukawa_4point(trial):
    maximum_occupation_number = int(
        np.random.choice(POSSIBLE_MAX_OCCUPATION_NUMBERS, size=1)
    )
    number_of_bosonic_modes = np.random.random_integers(2, MAX_NUMBER_OF_BOSONIC_MODES)
    number_of_fermionic_modes = np.random.random_integers(
        2, MAX_NUMBER_OF_FERMIONIC_MODES
    )
    bosonic_indices = list(
        np.random.choice(range(number_of_bosonic_modes), size=2, replace=False)
    )
    fermionic_indices = list(
        np.random.choice(range(number_of_fermionic_modes), size=2, replace=False)
    )

    operator_string = f"b{fermionic_indices[1]} b{fermionic_indices[0]} a{bosonic_indices[1]}^ a{bosonic_indices[0]}^"
    operator = ParticleOperator(operator_string)
    conjugate_string = f"b{fermionic_indices[0]}^ b{fermionic_indices[1]}^ a{bosonic_indices[1]} a{bosonic_indices[0]}"
    operator += ParticleOperator(conjugate_string)

    _validate_clean_ancillae_are_cleaned(
        operator,
        3,
        maximum_occupation_number,
        bosonic_indices + fermionic_indices,
        yukawa_4point_pair_term_block_encoding,
    )
    _validate_block_encoding_does_nothing_when_control_is_off(
        operator,
        3,
        maximum_occupation_number,
        bosonic_indices + fermionic_indices,
        yukawa_4point_pair_term_block_encoding,
    )
    _validate_block_encoding(
        maximum_occupation_number,
        operator,
        3,
        maximum_occupation_number,
        bosonic_indices + fermionic_indices,
        yukawa_4point_pair_term_block_encoding,
    )
