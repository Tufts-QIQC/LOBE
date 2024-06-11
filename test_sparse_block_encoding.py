import pytest
import cirq
import numpy as np
from usp import add_naive_usp
from coefficient_oracle import add_coefficient_oracle
from _utils import get_index_of_reversed_bitstring
from select_oracle import add_select_oracle


@pytest.mark.parametrize("number_of_operators", range(1, 17))
def test_naive_usp_prepares_correct_state(number_of_operators):
    number_of_qubits = int(np.ceil(np.log2(number_of_operators)))
    simulator = cirq.Simulator(dtype=np.complex128)

    circuit = cirq.Circuit()
    qubits = [cirq.LineQubit(i) for i in range(number_of_qubits)]

    circuit = add_naive_usp(circuit, qubits)

    wavefunction = simulator.simulate(circuit).final_state_vector

    expected_wavefunction = (1 / np.sqrt(1 << number_of_qubits)) * np.ones(
        1 << number_of_qubits
    )

    assert np.allclose(expected_wavefunction, wavefunction)


@pytest.mark.parametrize("number_of_operators", range(1, 17))
def test_coefficient_oracle_operates_on_individual_l_states_as_expected(
    number_of_operators,
):
    for state_of_index in range(number_of_operators):
        # state_of_index = np.random.choice(range(0, number_of_operators))
        term_coefficients = np.random.uniform(-100, 100, size=number_of_operators)
        hamiltonian_norm = sum(np.abs(term_coefficients))
        normalized_term_coefficients = term_coefficients / hamiltonian_norm

        number_of_index_qubits = int(np.ceil(np.log2(number_of_operators))) + (
            number_of_operators == 1
        )
        simulator = cirq.Simulator(dtype=np.complex128)
        circuit = cirq.Circuit()
        rotation_qubit = cirq.LineQubit(0)
        index_qubits = [cirq.LineQubit(i + 1) for i in range(number_of_index_qubits)]

        circuit = add_coefficient_oracle(
            circuit,
            rotation_qubit,
            index_qubits,
            normalized_term_coefficients,
            number_of_operators,
        )

        initial_index_state = np.zeros(1 << (number_of_index_qubits))
        initial_index_state[
            get_index_of_reversed_bitstring(state_of_index, number_of_index_qubits)
        ] = 1  # set amplitude
        initial_state = np.kron([1, 0], initial_index_state)  # |0> tensor |l>

        wavefunction = simulator.simulate(
            circuit, initial_state=initial_state
        ).final_state_vector

        expected_wavefunction = np.zeros(1 << (number_of_index_qubits + 1))
        expected_wavefunction = np.kron(
            [
                normalized_term_coefficients[state_of_index],
                np.sqrt(1 - normalized_term_coefficients[state_of_index] ** 2),
            ],
            initial_index_state,
        )
        assert np.allclose(wavefunction, expected_wavefunction)


@pytest.mark.parametrize("number_of_operators", range(1, 17))
def test_coefficient_oracle_in_superposition(number_of_operators):

    term_coefficients = np.random.uniform(-100, 100, size=number_of_operators)
    hamiltonian_norm = sum(np.abs(term_coefficients))
    normalized_term_coefficients = term_coefficients / hamiltonian_norm

    number_of_index_qubits = int(np.ceil(np.log2(number_of_operators))) + (
        number_of_operators == 1
    )
    simulator = cirq.Simulator(dtype=np.complex128)
    circuit = cirq.Circuit()
    rotation_qubit = cirq.LineQubit(0)
    index_qubits = [cirq.LineQubit(i + 1) for i in range(number_of_index_qubits)]

    circuit = add_coefficient_oracle(
        circuit,
        rotation_qubit,
        index_qubits,
        normalized_term_coefficients,
        number_of_operators,
    )

    initial_index_state_prep_random = (
        np.random.uniform(-1, 1, 1 << number_of_index_qubits)
        + np.random.uniform(-1, 1, 1 << number_of_index_qubits) * 1j
    )
    initial_index_state_prep_random /= np.linalg.norm(initial_index_state_prep_random)
    assert np.isclose(np.linalg.norm(initial_index_state_prep_random), 1)
    initial_state = np.kron([1, 0], initial_index_state_prep_random)  # |0> tensor |l>

    wavefunction = simulator.simulate(
        circuit, initial_state=initial_state
    ).final_state_vector

    expected_wavefunction = np.zeros(
        1 << (number_of_index_qubits + 1), dtype=np.complex128
    )
    for state_of_index in range(1 << number_of_index_qubits):
        new_state_of_index_integer_index = get_index_of_reversed_bitstring(
            state_of_index, number_of_index_qubits
        )
        alpha_coeff = initial_index_state_prep_random[new_state_of_index_integer_index]
        if state_of_index < number_of_operators:
            ham_coeff = normalized_term_coefficients[state_of_index]
            expected_wavefunction[new_state_of_index_integer_index] = (
                ham_coeff * alpha_coeff
            )
            expected_wavefunction[
                new_state_of_index_integer_index + (1 << number_of_index_qubits)
            ] = (np.sqrt(1 - ham_coeff**2) * alpha_coeff)
        else:
            expected_wavefunction[new_state_of_index_integer_index] = alpha_coeff

    assert np.isclose(np.linalg.norm(expected_wavefunction), 1)
    assert np.allclose(wavefunction, expected_wavefunction)


TOY_HAMILTONIAN_SELECT_STATE_MAP = {
    "1" + "0" + "00"[::-1] + "00"[::-1]: "1" + "0" + "00"[::-1] + "00"[::-1],
    "1" + "0" + "01"[::-1] + "00"[::-1]: "1" + "0" + "01"[::-1] + "00"[::-1],
    "1" + "0" + "10"[::-1] + "00"[::-1]: "1" + "0" + "10"[::-1] + "00"[::-1],
    "1" + "0" + "11"[::-1] + "00"[::-1]: "1" + "0" + "11"[::-1] + "00"[::-1],
    "1" + "0" + "00"[::-1] + "01"[::-1]: "0" + "0" + "00"[::-1] + "01"[::-1],
    "1" + "0" + "01"[::-1] + "01"[::-1]: "1" + "0" + "01"[::-1] + "01"[::-1],
    "1" + "0" + "10"[::-1] + "01"[::-1]: "0" + "0" + "10"[::-1] + "10"[::-1],
    "1" + "0" + "11"[::-1] + "01"[::-1]: "1" + "0" + "11"[::-1] + "01"[::-1],
    "1" + "0" + "00"[::-1] + "10"[::-1]: "1" + "0" + "00"[::-1] + "10"[::-1],
    "1" + "0" + "01"[::-1] + "10"[::-1]: "0" + "0" + "01"[::-1] + "01"[::-1],
    "1" + "0" + "10"[::-1] + "10"[::-1]: "1" + "0" + "10"[::-1] + "10"[::-1],
    "1" + "0" + "11"[::-1] + "10"[::-1]: "0" + "0" + "11"[::-1] + "10"[::-1],
    "1" + "0" + "00"[::-1] + "11"[::-1]: "0" + "0" + "00"[::-1] + "11"[::-1],
    "1" + "0" + "01"[::-1] + "11"[::-1]: "1" + "0" + "01"[::-1] + "11"[::-1],
    "1" + "0" + "10"[::-1] + "11"[::-1]: "1" + "0" + "10"[::-1] + "11"[::-1],
    "1" + "0" + "11"[::-1] + "11"[::-1]: "0" + "0" + "11"[::-1] + "11"[::-1],
}


def get_select_oracle_test_inputs():
    simulator = cirq.Simulator(dtype=np.complex128)
    number_of_index_qubits = 2
    operators = [(0, 0), (0, 1), (1, 0), (1, 1)]
    circuit = cirq.Circuit()
    validation = cirq.LineQubit(0)
    control = cirq.LineQubit(1)
    index = [cirq.LineQubit(i + 2) for i in range(2)]
    system = [cirq.LineQubit(i + 4) for i in range(2)]

    circuit = add_select_oracle(circuit, validation, control, index, system, operators)

    initial_state_of_validation = np.zeros(2)
    initial_state_of_validation[1] = 1  # |1>
    initial_state_of_control = np.zeros(2)
    initial_state_of_control[0] = 1  # |0>
    initial_state_of_validation_and_control = np.kron(
        initial_state_of_validation, initial_state_of_control
    )  # |1> tensor |0>

    intitial_state_of_index = (
        np.random.uniform(-1, 1, 1 << number_of_index_qubits)
        + np.random.uniform(-1, 1, 1 << number_of_index_qubits) * 1j
    )
    intitial_state_of_index /= np.linalg.norm(intitial_state_of_index)
    intitial_state_of_val_control_index = np.kron(
        initial_state_of_validation_and_control, intitial_state_of_index
    )

    return simulator, circuit, intitial_state_of_val_control_index


@pytest.mark.parametrize("system_basis_state", ["00", "01", "10", "11"])
def test_select_oracle_on_basis_state_for_toy_hamiltonian(system_basis_state):
    simulator, circuit, intitial_state_of_val_control_index = (
        get_select_oracle_test_inputs()
    )

    index_of_system_state = int(system_basis_state, 2)
    initial_state_of_system = np.zeros(4)
    initial_state_of_system[
        get_index_of_reversed_bitstring(index_of_system_state, 2)
    ] = 1
    initial_state = np.kron(
        intitial_state_of_val_control_index, initial_state_of_system
    )

    wavefunction = simulator.simulate(
        circuit, initial_state=initial_state
    ).final_state_vector

    for index_bitstring in ["00", "01", "10", "11"]:
        initial_bitstring = "1" + "0" + index_bitstring[::-1] + system_basis_state[::-1]
        assert initial_state[int(initial_bitstring, 2)] != 0
        assert np.isclose(
            initial_state[int(initial_bitstring, 2)],
            wavefunction[int(TOY_HAMILTONIAN_SELECT_STATE_MAP[initial_bitstring], 2)],
        )


def test_select_oracle_on_superposition_state_for_toy_hamiltonian():
    simulator, circuit, intitial_state_of_val_control_index = (
        get_select_oracle_test_inputs()
    )

    random_fock_state_coeffs = (
        np.random.uniform(-1, 1, size=4) + np.random.uniform(-1, 1, size=4) * 1j
    )
    random_fock_state_coeffs /= np.linalg.norm(random_fock_state_coeffs)

    initial_state_of_system = np.zeros(4, dtype=np.complex128)
    for system_state_index in range(4):
        initial_state_of_system[
            get_index_of_reversed_bitstring(system_state_index, 2)
        ] = random_fock_state_coeffs[system_state_index]
    initial_state = np.kron(
        intitial_state_of_val_control_index, initial_state_of_system
    )

    wavefunction = simulator.simulate(
        circuit, initial_state=initial_state
    ).final_state_vector

    for index_state in ["00", "01", "10", "11"]:
        for system_state in ["00", "01", "10", "11"]:
            initial_bitstring = (
                "1" + "0" + index_state[::-1] + system_state[::-1]
            )  # validation, control, index, system
            assert initial_state[int(initial_bitstring, 2)] != 0
            assert np.isclose(
                initial_state[int(initial_bitstring, 2)],
                wavefunction[
                    int(TOY_HAMILTONIAN_SELECT_STATE_MAP[initial_bitstring], 2)
                ],
            )


def test_select_oracle_block_encoding_for_toy_hamiltonian():
    operators = [(0, 0), (0, 1), (1, 0), (1, 1)]
    coefficients = [1, 1, 1, 1]
    circuit = cirq.Circuit()
    validation = cirq.LineQubit(0)
    control = cirq.LineQubit(1)
    rotation = cirq.LineQubit(2)
    index = [cirq.LineQubit(i + 3) for i in range(2)]
    system = [cirq.LineQubit(i + 5) for i in range(2)]

    circuit = add_naive_usp(circuit, index)
    circuit.append(cirq.X.on(validation))
    circuit = add_select_oracle(circuit, validation, control, index, system, operators)
    circuit = add_coefficient_oracle(circuit, rotation, index, coefficients, 4)
    circuit = add_naive_usp(circuit, index)

    upper_left_block = circuit.unitary()[: 1 << len(system), : 1 << len(system)]
    normalized_hamiltonian = (
        np.array(
            [
                np.array([0, 0, 0, 0]),
                np.array([0, 1, 1, 0]),
                np.array([0, 1, 1, 0]),
                np.array([0, 0, 0, 2]),
            ]
        )
        / 4
    )
    assert np.allclose(upper_left_block, normalized_hamiltonian)
