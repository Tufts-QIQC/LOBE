import pytest
import numpy as np
import cirq
from src.lobe.coefficient_oracle import add_coefficient_oracle
from src.lobe._utils import get_index_of_reversed_bitstring


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
