import pytest
import numpy as np
import cirq
from src.lobe.select_oracle import add_select_oracle


TOY_HAMILTONIAN_SELECT_STATE_MAP = {
    "1" + "0" + "00" + "00": "1" + "0" + "00" + "00",
    "1" + "0" + "01" + "00": "1" + "0" + "01" + "00",
    "1" + "0" + "10" + "00": "1" + "0" + "10" + "00",
    "1" + "0" + "11" + "00": "1" + "0" + "11" + "00",
    "1" + "0" + "00" + "01": "0" + "0" + "00" + "01",
    "1" + "0" + "01" + "01": "1" + "0" + "01" + "01",
    "1" + "0" + "10" + "01": "0" + "0" + "10" + "10",
    "1" + "0" + "11" + "01": "1" + "0" + "11" + "01",
    "1" + "0" + "00" + "10": "1" + "0" + "00" + "10",
    "1" + "0" + "01" + "10": "0" + "0" + "01" + "01",
    "1" + "0" + "10" + "10": "1" + "0" + "10" + "10",
    "1" + "0" + "11" + "10": "0" + "0" + "11" + "10",
    "1" + "0" + "00" + "11": "0" + "0" + "00" + "11",
    "1" + "0" + "01" + "11": "1" + "0" + "01" + "11",
    "1" + "0" + "10" + "11": "1" + "0" + "10" + "11",
    "1" + "0" + "11" + "11": "0" + "0" + "11" + "11",
}


def get_select_oracle_test_inputs():
    simulator = cirq.Simulator(dtype=np.complex128)
    number_of_index_qubits = 2
    operators = [((0, 0), (0, 0)), ((0, 0), (0, 1)), ((0, 1), (0, 0)), ((0, 1), (0, 1))]
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
@pytest.mark.parametrize("index_bitstring", ["00", "01", "10", "11"])
def test_select_oracle_on_basis_state_for_toy_hamiltonian(
    system_basis_state, index_bitstring
):
    simulator, circuit, intitial_state_of_val_control_index = (
        get_select_oracle_test_inputs()
    )

    index_of_system_state = int(system_basis_state, 2)
    initial_state_of_system = np.zeros(4)
    initial_state_of_system[index_of_system_state] = 1
    initial_state = np.kron(
        intitial_state_of_val_control_index, initial_state_of_system
    )

    wavefunction = simulator.simulate(
        circuit, initial_state=initial_state
    ).final_state_vector

    initial_bitstring = "1" + "0" + index_bitstring + system_basis_state
    assert initial_state[int(initial_bitstring, 2)] != 0
    assert np.isclose(
        initial_state[int(initial_bitstring, 2)],
        wavefunction[int(TOY_HAMILTONIAN_SELECT_STATE_MAP[initial_bitstring], 2)],
    )


@pytest.mark.parametrize("index_state", ["00", "01", "10", "11"])
@pytest.mark.parametrize("system_state", ["00", "01", "10", "11"])
def test_select_oracle_on_superposition_state_for_toy_hamiltonian(
    index_state, system_state
):
    simulator, circuit, intitial_state_of_val_control_index = (
        get_select_oracle_test_inputs()
    )

    random_fock_state_coeffs = (
        np.random.uniform(-1, 1, size=4) + np.random.uniform(-1, 1, size=4) * 1j
    )
    random_fock_state_coeffs /= np.linalg.norm(random_fock_state_coeffs)

    initial_state_of_system = np.zeros(4, dtype=np.complex128)
    for system_state_index in range(4):
        initial_state_of_system[system_state_index] = random_fock_state_coeffs[
            system_state_index
        ]
    initial_state = np.kron(
        intitial_state_of_val_control_index, initial_state_of_system
    )

    wavefunction = simulator.simulate(
        circuit, initial_state=initial_state
    ).final_state_vector

    initial_bitstring = (
        "1" + "0" + index_state + system_state
    )  # validation, control, index, system
    assert initial_state[int(initial_bitstring, 2)] != 0
    assert np.isclose(
        initial_state[int(initial_bitstring, 2)],
        wavefunction[int(TOY_HAMILTONIAN_SELECT_STATE_MAP[initial_bitstring], 2)],
    )


def test_select_oracle_on_one_two_body_term():
    """
    1. Create operator to test
    2. Create blank circuit
    3. Initialize state as |v>|c>|r>|l>|j> with |l> = |0>
    4. Append oracles
    5. Simulate
    6. Compare
    """
    # Operator = b_3^dag b_2^dag b_1 b_0
    # This acts only on |0, 0, 1, 1> to output |1, 1, 0, 0>
    operator = [((0, 3), (0, 2), (0, 1), (0, 0))]

    number_of_index_qubits = 1
    number_of_system_qubits = 4

    circuit = cirq.Circuit()

    validation = cirq.LineQubit(0)
    control = cirq.LineQubit(1)
    rotation = cirq.LineQubit(2)
    index_register = [cirq.LineQubit(i + 3) for i in range(number_of_index_qubits)]
    system_register = [
        cirq.LineQubit(i + 3 + number_of_index_qubits)
        for i in range(number_of_system_qubits)
    ]

    circuit.append(cirq.X.on(validation))
    circuit.append(cirq.I.on(rotation))
    circuit.append(cirq.I.on_each(*system_register))

    circuit = add_select_oracle(
        circuit, validation, control, index_register, system_register, operator
    )

    num_qubits = 3 + number_of_index_qubits + number_of_system_qubits

    all_registers_bar_j = np.zeros(1 << (num_qubits - number_of_system_qubits))
    all_registers_bar_j[0] = (
        1  # |000..0> corresponds to a one in the first slot of the array
    )

    init_j = np.zeros(2**number_of_system_qubits)  # |j> = |0011>
    j_str = "0011"
    init_j[int(j_str, 2)] = 1

    initial_state = np.kron(all_registers_bar_j, init_j)

    simulator = cirq.Simulator(dtype=np.complex128)

    wavefunction = simulator.simulate(
        circuit, initial_state=initial_state
    ).final_state_vector

    expected_all_registers_bar_j = np.zeros(2 ** (3 + number_of_index_qubits))
    expected_all_registers_bar_j[0] = 1  # |000> \otimes |0>

    expect_init_j = np.zeros(2**number_of_system_qubits)  # |j> = |0011>
    expect_j_str = "1100"
    expect_init_j[int(expect_j_str, 2)] = -1

    expected_final_wavefunction = np.kron(expected_all_registers_bar_j, expect_init_j)
    assert np.allclose(wavefunction, expected_final_wavefunction)


@pytest.mark.parametrize(
    ["j_str", "expect_j_str", "parity_coeff"],
    [("00011", "11000", -1), ("00111", "11100", -1)],
)
def test_parity_on_five_qubit_one_two_body_term(j_str, expect_j_str, parity_coeff):
    # b_4^dag b_3^dag b_1 b_0 |00011> = -|11000> & |00111> = -|11100>
    operator = [((0, 4), (0, 3), (0, 1), (0, 0))]

    number_of_index_qubits = 1
    number_of_system_qubits = 5

    circuit = cirq.Circuit()

    validation = cirq.LineQubit(0)
    control = cirq.LineQubit(1)
    rotation = cirq.LineQubit(2)
    index_register = [cirq.LineQubit(i + 3) for i in range(number_of_index_qubits)]
    system_register = [
        cirq.LineQubit(i + 3 + number_of_index_qubits)
        for i in range(number_of_system_qubits)
    ]

    circuit.append(cirq.X.on(validation))
    circuit.append(cirq.I.on(rotation))
    circuit.append(cirq.I.on_each(*system_register))

    circuit = add_select_oracle(
        circuit, validation, control, index_register, system_register, operator
    )

    num_qubits = 3 + number_of_index_qubits + number_of_system_qubits

    all_registers_bar_j = np.zeros(1 << (num_qubits - number_of_system_qubits))
    all_registers_bar_j[0] = (
        1  # |000..0> corresponds to a one in the first slot of the array
    )

    init_j = np.zeros(2**number_of_system_qubits)  # |j> = |init_j>
    init_j[int(j_str, 2)] = 1

    initial_state = np.kron(all_registers_bar_j, init_j)

    simulator = cirq.Simulator(dtype=np.complex128)

    wavefunction = simulator.simulate(
        circuit, initial_state=initial_state
    ).final_state_vector

    expected_all_registers_bar_j = np.zeros(2 ** (3 + number_of_index_qubits))
    expected_all_registers_bar_j[0] = 1  # |000> \otimes |0>

    expect_init_j = np.zeros(2**number_of_system_qubits)  # |j> = |expect_j_str>
    expect_init_j[int(expect_j_str, 2)] = parity_coeff * 1.0

    expected_final_wavefunction = np.kron(expected_all_registers_bar_j, expect_init_j)
    assert np.allclose(wavefunction, expected_final_wavefunction)


@pytest.mark.parametrize(
    ["j_str", "expect_j_str", "parity_coeff", "index_state"],
    [
        ("00011", "11000", -1, "0"),
        ("00111", "11100", -1, "0"),
        ("00010", "01000", 1, "1"),
        ("00011", "01001", 1, "1"),
        ("00111", "01101", -1, "1"),
        ("00110", "01100", -1, "1"),
    ],
)
def test_select_oracle_on_both_one_and_two_body_terms(
    j_str, expect_j_str, parity_coeff, index_state
):
    # b_4^dag b_3^dag b_1 b_0 |00011> = |11000> & |00111> = -|11100>
    operator = [((0, 4), (0, 3), (0, 1), (0, 0)), ((0, 3), (0, 1))]

    number_of_index_qubits = 1
    number_of_system_qubits = 5

    circuit = cirq.Circuit()

    validation = cirq.LineQubit(0)
    control = cirq.LineQubit(1)
    rotation = cirq.LineQubit(2)
    index_register = [cirq.LineQubit(i + 3) for i in range(number_of_index_qubits)]
    system_register = [
        cirq.LineQubit(i + 3 + number_of_index_qubits)
        for i in range(number_of_system_qubits)
    ]

    circuit.append(cirq.X.on(validation))
    circuit.append(cirq.I.on(rotation))
    circuit.append(cirq.I.on_each(*system_register))

    circuit = add_select_oracle(
        circuit, validation, control, index_register, system_register, operator
    )

    num_qubits = 3 + number_of_index_qubits + number_of_system_qubits

    all_registers_bar_j_and_l = np.zeros(
        1 << (num_qubits - number_of_system_qubits - number_of_index_qubits)
    )
    all_registers_bar_j_and_l[0] = (
        1  # |000> corresponds to a one in the first slot of the array
    )
    init_l = np.zeros(2**number_of_index_qubits)
    init_l[int(index_state, 2)] = 1  # |l> = |index_state>
    all_registers_bar_j = np.kron(all_registers_bar_j_and_l, init_l)
    init_j = np.zeros(2**number_of_system_qubits)  # |j> = |init_j>
    init_j[int(j_str, 2)] = 1

    initial_state = np.kron(all_registers_bar_j, init_j)

    simulator = cirq.Simulator(dtype=np.complex128)

    wavefunction = simulator.simulate(
        circuit, initial_state=initial_state
    ).final_state_vector

    expect_final_j = np.zeros(2**number_of_system_qubits)  # |j> = |expect_j_str>
    expect_final_j[int(expect_j_str, 2)] = parity_coeff * 1.0

    expected_final_wavefunction = np.kron(all_registers_bar_j, expect_final_j)
    assert np.allclose(wavefunction, expected_final_wavefunction)
