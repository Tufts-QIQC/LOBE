from openparticle import ParticleOperator, generate_matrix
import numpy as np
import cirq
from src.lobe.system import System
from src.lobe._utils import get_basis_of_full_system
from src.lobe.fermionic import (
    fermionic_plus_hc_block_encoding,
    fermionic_product_block_encoding,
)
import pytest
import math


def _validate_block_encoding(
    operator, active_indices, operator_types, block_encoding_function
):
    number_of_modes = max([term.max_mode for term in operator.to_list()]) + 1

    number_of_clean_ancillae = 100

    # Declare Qubits
    circuit = cirq.Circuit()
    control = cirq.LineQubit(0)
    block_encoding_ancilla = cirq.LineQubit(1)

    clean_ancillae = [cirq.LineQubit(i + 2) for i in range(number_of_clean_ancillae)]
    system = System(
        number_of_modes=number_of_modes,
        maximum_occupation_number=1,
        number_of_used_qubits=2 + number_of_clean_ancillae,
        has_fermions=True,
    )
    circuit.append(
        cirq.I.on_each(
            control,
            block_encoding_ancilla,
            *system.fermionic_register,
        )
    )

    if len(circuit.all_qubits()) >= 32:
        pytest.skip(f"too many qubits to build circuit: {len(circuit.all_qubits())}")

    # Flip control qubit so that we can focus on the 0-subspace of the control
    circuit.append(cirq.X.on(control))
    # Generate full Block-Encoding circuit
    gates, metrics = block_encoding_function(
        system,
        block_encoding_ancilla,
        active_indices,
        operator_types,
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
            1 << (len(circuit.all_qubits()) - system.number_of_system_qubits - 2),
            dtype=complex,
        )
        zero_state[0] = 1
        initial_control_state = [1, 0]
        initial_block_encoding_ancilla_state = [1, 0]
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
            1,
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
            1,
            has_fermions=operator.has_fermions,
            has_antifermions=operator.has_antifermions,
            has_bosons=operator.has_bosons,
        )
        matrix = generate_matrix(operator, full_fock_basis)

        upper_left_block = circuit.unitary(dtype=complex)[
            : 1 << system.number_of_system_qubits, : 1 << system.number_of_system_qubits
        ]

        assert np.allclose(upper_left_block, matrix)


def _validate_clean_ancillae_are_cleaned(
    operator, active_indices, operator_types, block_encoding_function
):
    number_of_modes = max([term.max_mode for term in operator.to_list()]) + 1

    number_of_clean_ancillae = 100

    # Declare Qubits
    circuit = cirq.Circuit()
    control = cirq.LineQubit(0)
    block_encoding_ancilla = cirq.LineQubit(1)

    clean_ancillae = [cirq.LineQubit(i + 2) for i in range(number_of_clean_ancillae)]
    system = System(
        number_of_modes=number_of_modes,
        maximum_occupation_number=1,
        number_of_used_qubits=2 + number_of_clean_ancillae,
        has_fermions=operator.has_fermions,
        has_antifermions=operator.has_antifermions,
        has_bosons=operator.has_bosons,
    )
    circuit.append(
        cirq.I.on_each(
            control,
            block_encoding_ancilla,
            *system.fermionic_register,
        )
    )

    if len(circuit.all_qubits()) >= 32:
        pytest.skip(f"too many qubits to build circuit: {len(circuit.all_qubits())}")

    # Flip control qubit so that we can focus on the 0-subspace of the control
    circuit.append(cirq.X.on(control))
    # Generate full Block-Encoding circuit
    gates, metrics = block_encoding_function(
        system,
        block_encoding_ancilla,
        active_indices,
        operator_types,
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
            1 << (len(circuit.all_qubits()) - system.number_of_system_qubits - 2),
            dtype=complex,
        )
        zero_state[0] = 1
        initial_control_state = 1j * np.random.uniform(-1, 1, 2) + np.random.uniform(
            -1, 1, 2
        )
        initial_control_state = initial_control_state / np.linalg.norm(
            initial_control_state
        )
        initial_block_encoding_ancilla_state = 1j * np.random.uniform(-1, 1, 2)
        initial_block_encoding_ancilla_state += np.random.uniform(-1, 1, 2)
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
            for bit in bitstring[2 : -system.number_of_system_qubits]:
                if bit == "1":
                    assert False


def _validate_block_encoding_does_nothing_when_control_is_off(
    operator, active_indices, operator_types, block_encoding_function
):
    number_of_modes = max([term.max_mode for term in operator.to_list()]) + 1

    number_of_clean_ancillae = 100

    # Declare Qubits
    circuit = cirq.Circuit()
    control = cirq.LineQubit(0)
    block_encoding_ancilla = cirq.LineQubit(1)

    clean_ancillae = [cirq.LineQubit(i + 2) for i in range(number_of_clean_ancillae)]
    system = System(
        number_of_modes=number_of_modes,
        maximum_occupation_number=1,
        number_of_used_qubits=2 + number_of_clean_ancillae,
        has_fermions=operator.has_fermions,
        has_antifermions=operator.has_antifermions,
        has_bosons=operator.has_bosons,
    )
    circuit.append(
        cirq.I.on_each(
            control,
            block_encoding_ancilla,
            *system.fermionic_register,
        )
    )

    if len(circuit.all_qubits()) >= 32:
        pytest.skip(f"too many qubits to build circuit: {len(circuit.all_qubits())}")

    # Generate full Block-Encoding circuit
    gates, metrics = block_encoding_function(
        system,
        block_encoding_ancilla,
        active_indices,
        operator_types,
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
            1 << (len(circuit.all_qubits()) - system.number_of_system_qubits - 2),
            dtype=complex,
        )
        zero_state[0] = 1
        initial_control_state = [1, 0]
        initial_block_encoding_ancilla_state = [1, 0]
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


def _check_numerics(operator, active_indices, operator_types, block_encoding_function):
    number_of_modes = max([term.max_mode for term in operator.to_list()]) + 1

    number_of_clean_ancillae = 100

    # Declare Qubits
    circuit = cirq.Circuit()
    control = cirq.LineQubit(0)
    block_encoding_ancilla = cirq.LineQubit(1)

    clean_ancillae = [cirq.LineQubit(i + 2) for i in range(number_of_clean_ancillae)]
    system = System(
        number_of_modes=number_of_modes,
        maximum_occupation_number=1,
        number_of_used_qubits=2 + number_of_clean_ancillae,
        has_fermions=True,
    )
    circuit.append(
        cirq.I.on_each(
            control,
            block_encoding_ancilla,
            *system.fermionic_register,
        )
    )

    if len(circuit.all_qubits()) >= 32:
        pytest.skip(f"too many qubits to build circuit: {len(circuit.all_qubits())}")

    # Generate full Block-Encoding circuit
    _, metrics = block_encoding_function(
        system,
        block_encoding_ancilla,
        active_indices,
        operator_types,
        clean_ancillae=clean_ancillae,
        ctrls=([control], [1]),
    )

    number_of_number_ops = operator_types.count(2)

    assert metrics.number_of_elbows == len(active_indices) - 1
    assert metrics.clean_ancillae_usage[-1] == 0
    assert (
        max(metrics.clean_ancillae_usage)
        == (len(active_indices) - 2)  # elbows for qbool
        + 1  # elbow to apply toff that flips ancilla
    )


def _check_numerics_plus_hc(
    operator, active_indices, operator_types, block_encoding_function
):
    number_of_modes = max([term.max_mode for term in operator.to_list()]) + 1

    number_of_clean_ancillae = 100

    # Declare Qubits
    circuit = cirq.Circuit()
    control = cirq.LineQubit(0)
    block_encoding_ancilla = cirq.LineQubit(1)

    clean_ancillae = [cirq.LineQubit(i + 2) for i in range(number_of_clean_ancillae)]
    system = System(
        number_of_modes=number_of_modes,
        maximum_occupation_number=1,
        number_of_used_qubits=2 + number_of_clean_ancillae,
        has_fermions=True,
    )
    circuit.append(
        cirq.I.on_each(
            control,
            block_encoding_ancilla,
            *system.fermionic_register,
        )
    )

    if len(circuit.all_qubits()) >= 32:
        pytest.skip(f"too many qubits to build circuit: {len(circuit.all_qubits())}")

    # Generate full Block-Encoding circuit
    _, metrics = block_encoding_function(
        system,
        block_encoding_ancilla,
        active_indices,
        operator_types,
        clean_ancillae=clean_ancillae,
        ctrls=([control], [1]),
    )

    number_of_number_ops = operator_types.count(2)

    assert metrics.number_of_elbows == len(active_indices) - 1
    assert metrics.clean_ancillae_usage[-1] == 0
    assert (
        max(metrics.clean_ancillae_usage)
        == (len(active_indices) - number_of_number_ops - 1)  # parity qubits
        + (len(active_indices) - 2)  # elbows for qbool
        + 1  # elbow to apply toff that flips ancilla
    )


MAX_MODES = 7
MAX_ACTIVE_MODES = 7
MIN_ACTIVE_MODES = 2


@pytest.mark.parametrize("trial", range(100))
def test_arbitrary_fermionic_operator_with_hc(trial):
    number_of_active_modes = np.random.random_integers(
        MIN_ACTIVE_MODES, MAX_ACTIVE_MODES
    )
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

    operator_string = ""
    for mode, operator_type in zip(active_modes, operator_types_reversed):
        if operator_type == 0:
            operator_string += f" b{mode}"
        if operator_type == 1:
            operator_string += f" b{mode}^"
        if operator_type == 2:
            operator_string += f" b{mode}^ b{mode}"

    conjugate_operator_string = ""
    for mode, operator_type in zip(active_modes[::-1], operator_types_reversed[::-1]):
        if operator_type == 0:
            conjugate_operator_string += f" b{mode}^"
        if operator_type == 1:
            conjugate_operator_string += f" b{mode}"
        if operator_type == 2:
            conjugate_operator_string += f" b{mode}^ b{mode}"

    operator = ParticleOperator(operator_string) + ParticleOperator(
        conjugate_operator_string
    )

    _validate_clean_ancillae_are_cleaned(
        operator,
        active_modes[::-1],
        operator_types_reversed[::-1],
        fermionic_plus_hc_block_encoding,
    )
    _validate_block_encoding_does_nothing_when_control_is_off(
        operator,
        active_modes[::-1],
        operator_types_reversed[::-1],
        fermionic_plus_hc_block_encoding,
    )
    _validate_block_encoding(
        operator,
        active_modes[::-1],
        operator_types_reversed[::-1],
        fermionic_plus_hc_block_encoding,
    )
    _check_numerics_plus_hc(
        operator,
        active_modes[::-1],
        operator_types_reversed[::-1],
        fermionic_plus_hc_block_encoding,
    )


@pytest.mark.parametrize("trial", range(100))
def test_arbitrary_fermionic_product(trial):
    number_of_active_modes = np.random.random_integers(
        MIN_ACTIVE_MODES, MAX_ACTIVE_MODES
    )
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

    operator_string = ""
    for mode, operator_type in zip(active_modes, operator_types_reversed):
        if operator_type == 0:
            operator_string += f" b{mode}"
        if operator_type == 1:
            operator_string += f" b{mode}^"
        if operator_type == 2:
            operator_string += f" b{mode}^ b{mode}"

    operator = ParticleOperator(operator_string)

    _validate_clean_ancillae_are_cleaned(
        operator,
        active_modes[::-1],
        operator_types_reversed[::-1],
        fermionic_product_block_encoding,
    )
    _validate_block_encoding_does_nothing_when_control_is_off(
        operator,
        active_modes[::-1],
        operator_types_reversed[::-1],
        fermionic_product_block_encoding,
    )
    _validate_block_encoding(
        operator,
        active_modes[::-1],
        operator_types_reversed[::-1],
        fermionic_product_block_encoding,
    )
    _check_numerics(
        operator,
        active_modes[::-1],
        operator_types_reversed[::-1],
        fermionic_product_block_encoding,
    )
