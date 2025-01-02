from openparticle import ParticleOperator, generate_matrix
import numpy as np
import cirq
from src.lobe.system import System
from src.lobe._utils import get_basis_of_full_system
from src.lobe.bosonic import (
    bosonic_mode_block_encoding,
    bosonic_mode_plus_hc_block_encoding,
)
import pytest
from functools import partial


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
        cirq.LineQubit(i + (1 + number_of_block_encoding_ancillae))
        for i in range(number_of_clean_ancillae)
    ]
    system = System(
        number_of_modes=number_of_modes,
        maximum_occupation_number=maximum_occupation_number,
        number_of_used_qubits=(1 + number_of_block_encoding_ancillae)
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
    if number_of_block_encoding_ancillae == 1:
        block_encoding_ancillae = block_encoding_ancillae[0]
    gates, metrics = block_encoding_function(
        system,
        block_encoding_ancillae,
        clean_ancillae=clean_ancillae,
        ctrls=([control], [1]),
    )
    circuit += gates
    # Flip control qubit so that we can focus on the 0-subspace of the control
    circuit.append(cirq.X.on(control))

    if len(circuit.all_qubits()) >= 20:
        pytest.skip(f"too many qubits to validate: {len(circuit.all_qubits())}")

    return circuit, metrics, system


def _validate_block_encoding(
    circuit,
    system,
    expected_rescaling_factor,
    operator,
    maximum_occupation_number,
):

    if len(circuit.all_qubits()) >= 15:
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
            1 << (len(circuit.all_qubits()) - system.number_of_system_qubits - 1 - 1),
            dtype=complex,
        )
        zero_state[0] = 1
        initial_control_state = [1, 0]
        initial_block_encoding_ancilla_state = np.zeros(1 << 1)
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
            system.number_of_modes,
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
    circuit,
    system,
    number_of_block_encoding_ancillae,
):
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
            - (1 + number_of_block_encoding_ancillae)
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
    circuit,
    system,
    number_of_block_encoding_ancillae,
):
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
            - (1 + number_of_block_encoding_ancillae)
        ),
        dtype=complex,
    )
    zero_state[0] = 1
    initial_control_state = [0, 1]
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

    assert np.allclose(initial_state, output_state, atol=1e-6)


@pytest.mark.parametrize("number_of_modes", range(2, 4))
@pytest.mark.parametrize("active_mode", range(0, 4))
@pytest.mark.parametrize("maximum_occupation_number", [1, 3, 7])
@pytest.mark.parametrize("R", range(1, 4))
@pytest.mark.parametrize("S", range(1, 4))
def test_bosonic_mode_block_encoding(
    number_of_modes, active_mode, maximum_occupation_number, R, S
):
    if (maximum_occupation_number == 7) and (active_mode > 0):
        pytest.skip()

    active_mode = active_mode % number_of_modes
    operator = ParticleOperator(f"a{active_mode}^") ** R
    operator *= ParticleOperator(f"a{active_mode}") ** S
    expected_rescaling_factor = np.sqrt(maximum_occupation_number) ** (R + S)

    circuit, metrics, system = _setup(
        1,
        operator,
        maximum_occupation_number,
        partial(
            bosonic_mode_block_encoding, active_index=active_mode, exponents=(R, S)
        ),
    )
    _validate_block_encoding(
        circuit,
        system,
        expected_rescaling_factor,
        operator,
        maximum_occupation_number,
    )
    _validate_clean_ancillae_are_cleaned(
        circuit,
        system,
        1,
    )
    _validate_block_encoding_does_nothing_when_control_is_off(circuit, system, 1)
    assert metrics.number_of_elbows <= maximum_occupation_number + 1 + max(
        int(np.log2(maximum_occupation_number + 1)) - 1, 0
    )
    assert metrics.number_of_rotations <= maximum_occupation_number + 2
    assert metrics.clean_ancillae_usage[-1] == 0


@pytest.mark.parametrize("number_of_modes", range(2, 4))
@pytest.mark.parametrize("active_mode", range(0, 4))
@pytest.mark.parametrize("maximum_occupation_number", [1, 3, 7])
@pytest.mark.parametrize("R", range(1, 4))
@pytest.mark.parametrize("S", range(1, 4))
def test_bosonic_mode_plus_hc_block_encoding(
    number_of_modes, active_mode, maximum_occupation_number, R, S
):
    if R == S:
        pytest.skip()
    if (maximum_occupation_number == 7) and (active_mode > 0):
        pytest.skip()

    active_mode = active_mode % number_of_modes
    operator = ParticleOperator(f"a{active_mode}^") ** R
    operator *= ParticleOperator(f"a{active_mode}") ** S
    hc_operator = ParticleOperator(f"a{active_mode}^") ** S
    hc_operator *= ParticleOperator(f"a{active_mode}") ** R
    operator += hc_operator
    expected_rescaling_factor = 2 * np.sqrt(maximum_occupation_number) ** (R + S)

    circuit, metrics, system = _setup(
        2,
        operator,
        maximum_occupation_number,
        partial(
            bosonic_mode_plus_hc_block_encoding,
            active_index=active_mode,
            exponents=(R, S),
        ),
    )
    _validate_block_encoding(
        circuit,
        system,
        expected_rescaling_factor,
        operator,
        maximum_occupation_number,
    )
    _validate_clean_ancillae_are_cleaned(
        circuit,
        system,
        2,
    )
    _validate_block_encoding_does_nothing_when_control_is_off(circuit, system, 2)
    assert metrics.number_of_elbows <= max(
        maximum_occupation_number + (2 * int(np.log2(maximum_occupation_number + 1))), 0
    )
    assert metrics.number_of_rotations <= maximum_occupation_number + 2
    assert metrics.clean_ancillae_usage[-1] == 0
