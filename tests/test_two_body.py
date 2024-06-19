import pytest
import numpy as np
import cirq
from src.lobe.toy_two_body_select_oracle import add__toy_pairing_select_oracle


def test_select_oracle_on_one_two_body_term():

    '''
    1. Create operator to test
    2. Create blank circuit
    3. Initialize state as |v>|c>|r>|l>|j> with |l> = |0>
    4. Append oracles
    5. Simulate
    6. Compare
    '''
    #Operator = b_3^dag b_2^dag b_1 b_0
    #This acts only on |0, 0, 1, 1> to output |1, 1, 0, 0>
    operator = [(3, 2, 1, 0)]

    number_of_index_qubits = 1
    number_of_system_qubits = 4

    circuit = cirq.Circuit()

    validation = cirq.LineQubit(0)
    control = cirq.LineQubit(1)
    rotation = cirq.LineQubit(2)
    index_register = [cirq.LineQubit(i + 3) for i in range(number_of_index_qubits)]
    system_register = [cirq.LineQubit(i + 3 + number_of_index_qubits) for i in range(number_of_system_qubits)]

    circuit = cirq.Circuit()

    circuit.append(cirq.X.on(validation))
    circuit.append(cirq.I.on(rotation))
    # circuit = add_naive_usp(circuit, index_register=index_register)

    circuit = add__toy_pairing_select_oracle(circuit, validation, control, index_register, system_register, operator)

    # circuit = add_naive_usp(circuit, index_register)

    num_qubits = 3 + number_of_index_qubits + number_of_system_qubits

    all_registers_bar_j = np.zeros(1 << (num_qubits - number_of_system_qubits))
    all_registers_bar_j[0] = 1 #|000..0> corresponds to a one in the first slot of the array

    init_j = np.zeros(2**number_of_system_qubits) #|j> = |0011>
    j_str = '0011'
    init_j[int(j_str, 2)] = 1

    initial_state = np.kron(all_registers_bar_j, init_j)

    simulator = cirq.Simulator(dtype=np.complex128)

    wavefunction = simulator.simulate(
            circuit, initial_state=initial_state
        ).final_state_vector
    
    expected_all_registers_bar_j = np.zeros(2**(3 + number_of_index_qubits))
    expected_all_registers_bar_j[0] = 1 #|000> \otimes |0>

    expect_init_j = np.zeros(2**number_of_system_qubits) #|j> = |0011>
    expect_j_str = '1100'
    expect_init_j[int(expect_j_str, 2)] = 1

    expected_final_wavefunction = np.kron(expected_all_registers_bar_j, expect_init_j)
    assert np.allclose(wavefunction, expected_final_wavefunction)

    

