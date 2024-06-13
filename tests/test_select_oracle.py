import pytest
import numpy as np
import cirq
from src.lobe.select_oracle import add_select_oracle
from src.lobe._utils import get_index_of_reversed_bitstring

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
