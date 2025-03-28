import cirq
import pytest
import numpy as np
from openparticle import generate_matrix
from src.lobe.system import System
from src.lobe._utils import get_basis_of_full_system


def _setup(
    number_of_block_encoding_ancillae,
    operator,
    maximum_occupation_number,
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
    number_of_fermionic_modes = 0
    number_of_bosonic_modes = 0
    if operator.max_fermionic_mode is not None:
        number_of_fermionic_modes = operator.max_fermionic_mode + 1
    if operator.max_bosonic_mode is not None:
        number_of_bosonic_modes = operator.max_bosonic_mode + 1
    system = System(
        maximum_occupation_number,
        1 + number_of_block_encoding_ancillae + number_of_clean_ancillae,
        number_of_fermionic_modes=number_of_fermionic_modes,
        number_of_bosonic_modes=number_of_bosonic_modes,
    )
    circuit.append(
        cirq.I.on_each(
            control,
            *block_encoding_ancillae,
            *system.fermionic_modes,
        )
    )
    for bosonic_reg in system.bosonic_modes:
        circuit.append(cirq.I.on_each(*bosonic_reg))

    if len(circuit.all_qubits()) >= 32:
        pytest.skip(f"too many qubits to build circuit: {len(circuit.all_qubits())}")

    # Flip control qubit so that we can focus on the 0-subspace of the control
    circuit.append(cirq.X.on(control))
    # Generate full Block-Encoding circuit
    gates, metrics = block_encoding_function(
        system,
        block_encoding_ancillae,
        clean_ancillae=clean_ancillae,
        ctrls=([control], [1]),
    )
    circuit += gates
    # Flip control qubit so that we can focus on the 0-subspace of the control
    circuit.append(cirq.X.on(control))

    if len(circuit.all_qubits()) >= 18:
        pytest.skip(f"too many qubits to validate: {len(circuit.all_qubits())}")

    return circuit, metrics, system


def _validate_block_encoding(
    circuit,
    system,
    expected_rescaling_factor,
    operator,
    number_of_block_encoding_ancillae,
    maximum_occupation_number,
    max_qubits=18,
    using_pytest=True,
):
    if len(circuit.all_qubits()) >= max_qubits:
        if using_pytest:
            pytest.skip(f"Too many qubits to validate: {len(circuit.all_qubits())}")
        else:
            print(f"Too many qubits to validate: {len(circuit.all_qubits())}")
    elif len(circuit.all_qubits()) >= 12:
        print(
            f"Testing singular quantum state for circuit with {len(circuit.all_qubits())} qubits"
        )
        simulator = cirq.Simulator()

        number_of_fermionic_modes = 0
        number_of_bosonic_modes = 0
        if operator.max_fermionic_mode is not None:
            number_of_fermionic_modes = operator.max_fermionic_mode + 1
        if operator.max_bosonic_mode is not None:
            number_of_bosonic_modes = operator.max_bosonic_mode + 1
        full_fock_basis = get_basis_of_full_system(
            maximum_occupation_number,
            number_of_fermionic_modes=number_of_fermionic_modes,
            number_of_bosonic_modes=number_of_bosonic_modes,
        )
        matrix = generate_matrix(operator, full_fock_basis)

        random_system_state = np.zeros(1 << system.number_of_system_qubits)
        attempts = 0
        while np.isclose(np.linalg.norm(matrix @ random_system_state), 0):
            random_system_state = 1j * np.random.uniform(
                -1, 1, 1 << system.number_of_system_qubits
            )
            random_system_state += np.random.uniform(
                -1, 1, 1 << system.number_of_system_qubits
            )
            random_system_state = random_system_state / np.linalg.norm(
                random_system_state
            )
            if attempts > 100:
                break
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

        expected_final_state = matrix @ random_system_state
        expected_final_state = expected_final_state / np.linalg.norm(
            expected_final_state
        )
        normalized_final_state = final_state / np.linalg.norm(final_state)
        squared_overlap = (
            np.abs(np.dot(expected_final_state.T.conj(), normalized_final_state)) ** 2
        )
        if (not using_pytest) and (not np.allclose(1, squared_overlap, atol=1e-2)):
            print(squared_overlap)
        assert np.allclose(1, squared_overlap, atol=1e-1)
    else:
        number_of_fermionic_modes = 0
        number_of_bosonic_modes = 0
        if operator.max_fermionic_mode is not None:
            number_of_fermionic_modes = operator.max_fermionic_mode + 1
        if operator.max_bosonic_mode is not None:
            number_of_bosonic_modes = operator.max_bosonic_mode + 1
        full_fock_basis = get_basis_of_full_system(
            maximum_occupation_number,
            number_of_fermionic_modes=number_of_fermionic_modes,
            number_of_bosonic_modes=number_of_bosonic_modes,
        )
        matrix = generate_matrix(operator, full_fock_basis)

        upper_left_block = circuit.unitary(dtype=complex)[
            : 1 << system.number_of_system_qubits, : 1 << system.number_of_system_qubits
        ]

        rescaled_upper_left_block = expected_rescaling_factor * upper_left_block

        if (not using_pytest) and (not np.allclose(rescaled_upper_left_block, matrix)):
            print(rescaled_upper_left_block.real.round(2))
            print(matrix.real.round(2))
        assert np.allclose(rescaled_upper_left_block, matrix)


def _validate_clean_ancillae_are_cleaned(
    circuit,
    system,
    number_of_block_encoding_ancillae,
    max_qubits=18,
    using_pytest=True,
):
    if len(circuit.all_qubits()) >= max_qubits:
        if using_pytest:
            pytest.skip(f"Too many qubits to validate: {len(circuit.all_qubits())}")
        else:
            print(f"Too many qubits to validate: {len(circuit.all_qubits())}")

    simulator = cirq.Simulator()
    random_system_state = 1j * np.random.uniform(
        -1, 1, 1 << system.number_of_system_qubits
    )
    random_system_state += np.random.uniform(-1, 1, 1 << system.number_of_system_qubits)
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
    if number_of_block_encoding_ancillae == 0:
        initial_block_encoding_ancilla_state = [1]
    else:
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
    circuit,
    system,
    number_of_block_encoding_ancillae,
    max_qubits=18,
    using_pytest=True,
):
    if len(circuit.all_qubits()) >= max_qubits:
        if using_pytest:
            pytest.skip(f"Too many qubits to validate: {len(circuit.all_qubits())}")
        else:
            print(f"Too many qubits to validate: {len(circuit.all_qubits())}")

    simulator = cirq.Simulator()
    random_system_state = 1j * np.random.uniform(
        -1, 1, 1 << system.number_of_system_qubits
    )
    random_system_state += np.random.uniform(-1, 1, 1 << system.number_of_system_qubits)
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
    initial_control_state = [
        0,
        1,
    ]  # Control being "off" is in the |1> state due to X gates in _setup
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

    assert np.allclose(output_state, initial_state, atol=1e-6)
