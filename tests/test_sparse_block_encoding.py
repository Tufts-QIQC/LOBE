import pytest
import cirq
import numpy as np
from src.lobe.usp import add_naive_usp
from src.lobe.coefficient_oracle import add_coefficient_oracle
from src.lobe.select_oracle import add_select_oracle


@pytest.mark.parametrize(
    ["operators", "coefficients", "hamiltonian"],
    [
        (
            [(0, 0), (0, 1), (1, 0), (1, 1)],
            np.array([1, 1, 1, 1]),
            np.array(
                [
                    np.array([0, 0, 0, 0]),
                    np.array([0, 1, 1, 0]),
                    np.array([0, 1, 1, 0]),
                    np.array([0, 0, 0, 2]),
                ]
            ),
        ),
        (
            [(0, 0), (0, 1), (1, 0), (1, 1)],
            np.array([1, 0.5, 0.5, 1]),
            np.array(
                [
                    np.array([0, 0, 0, 0]),
                    np.array([0, 1, 0.5, 0]),
                    np.array([0, 0.5, 1, 0]),
                    np.array([0, 0, 0, 2]),
                ]
            ),
        ),
        (
            [(0, 0), (1, 1), (2, 2)],
            np.array([4, 2, 1]),
            np.array(
                [
                    np.array([0, 0, 0, 0, 0, 0, 0, 0]),
                    np.array([0, 4, 0, 0, 0, 0, 0, 0]),
                    np.array([0, 0, 2, 0, 0, 0, 0, 0]),
                    np.array([0, 0, 0, 6, 0, 0, 0, 0]),
                    np.array([0, 0, 0, 0, 1, 0, 0, 0]),
                    np.array([0, 0, 0, 0, 0, 5, 0, 0]),
                    np.array([0, 0, 0, 0, 0, 0, 3, 0]),
                    np.array([0, 0, 0, 0, 0, 0, 0, 7]),
                ]
            ),
        ),
        (
            [(0, 0), (0, 1), (0, 2), (0, 3),
             (1, 0), (1, 1), (1, 2), (1, 3),
             (2, 0), (2, 1), (2, 2), (2, 3),
             (3, 0), (3, 1), (3, 2), (3, 3)],
            np.array([ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 ]),
            np.array(
                [
                    [    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0 ],
                    [    0,   1,   2,   0,   3,   0,   0,   0,   4,   0,   0,   0,   0,   0,   0,   0 ],
                    [    0,   5,   6,   0,   7,   0,   0,   0,   8,   0,   0,   0,   0,   0,   0,   0 ],
                    [    0,   0,   0,   7,   0,   7,  -3,   0,   0,   8,  -4,   0,   0,   0,   0,   0 ],
                    [    0,   9,  10,   0,  11,   0,   0,   0,  12,   0,   0,   0,   0,   0,   0,   0 ],
                    [    0,   0,   0,  10,   0,  12,   2,   0,   0,  12,   0,   0,  -4,   0,   0,   0 ],
                    [    0,   0,   0,  -9,   0,   5,  17,   0,   0,   0,  12,   0,  -8,   0,   0,   0 ],
                    [    0,   0,   0,   0,   0,   0,   0,  18,   0,   0,   0,  12,   0,  -8,   4,   0 ],
                    [    0,  13,  14,   0,  15,   0,   0,   0,  16,   0,   0,   0,   0,   0,   0,   0 ],
                    [    0,   0,   0,  14,   0,  15,   0,   0,   0,  17,   2,   0,   3,   0,   0,   0 ],
                    [    0,   0,   0, -13,   0,   0,  15,   0,   0,   5,  22,   0,   7,   0,   0,   0 ],
                    [    0,   0,   0,   0,   0,   0,   0,  15,   0,   0,   0,  23,   0,   7,  -3,   0 ],
                    [    0,   0,   0,   0,   0, -13, -14,   0,   0,   9,  10,   0,  27,   0,   0,   0 ],
                    [    0,   0,   0,   0,   0,   0,   0, -14,   0,   0,   0,  10,   0,  28,   2,   0 ],
                    [    0,   0,   0,   0,   0,   0,   0,  13,   0,   0,   0,  -9,   0,   5,  33,   0 ],
                    [    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  34 ],
                ]
            ),
        ),
    ],
)
def test_block_encoding_for_toy_hamiltonian(operators, coefficients, hamiltonian):
    size_of_system = max(max(operators)) + 1
    number_of_index_qubits = int(np.ceil(np.log2(len(operators))))
    circuit = cirq.Circuit()
    validation = cirq.LineQubit(0)
    control = cirq.LineQubit(1)
    rotation = cirq.LineQubit(2)
    index = [cirq.LineQubit(i + 3) for i in range(number_of_index_qubits)]
    system = [
        cirq.LineQubit(i + 3 + number_of_index_qubits) for i in range(size_of_system)
    ]
    normalization_factor = max(coefficients)
    normalized_coefficients = coefficients / normalization_factor

    circuit = add_naive_usp(circuit, index)
    circuit.append(cirq.X.on(validation))
    circuit = add_select_oracle(circuit, validation, control, index, system, operators)
    circuit = add_coefficient_oracle(
        circuit, rotation, index, normalized_coefficients, len(operators)
    )
    circuit = add_naive_usp(circuit, index)

    upper_left_block = circuit.unitary()[: 1 << size_of_system, : 1 << size_of_system]
    normalized_hamiltonian = hamiltonian / (
        (1 << number_of_index_qubits) * normalization_factor
    )
    assert np.allclose(upper_left_block, normalized_hamiltonian)


def test_select_and_coefficient_oracles_commute():
    operators = [(0, 0), (0, 1), (1, 0), (1, 1)]
    coefficients = np.random.uniform(0, 1, size=4)
    validation = cirq.LineQubit(0)
    control = cirq.LineQubit(1)
    rotation = cirq.LineQubit(2)
    index = [cirq.LineQubit(i + 3) for i in range(2)]
    system = [cirq.LineQubit(i + 5) for i in range(2)]

    circuit = cirq.Circuit()
    circuit = add_naive_usp(circuit, index)
    circuit.append(cirq.X.on(validation))
    circuit = add_select_oracle(circuit, validation, control, index, system, operators)
    circuit = add_coefficient_oracle(
        circuit, rotation, index, coefficients, len(operators)
    )
    circuit = add_naive_usp(circuit, index)
    unitary_select_first = circuit.unitary()

    circuit = cirq.Circuit()
    circuit = add_naive_usp(circuit, index)
    circuit.append(cirq.X.on(validation))
    circuit = add_coefficient_oracle(
        circuit, rotation, index, coefficients, len(operators)
    )
    circuit = add_select_oracle(circuit, validation, control, index, system, operators)
    circuit = add_naive_usp(circuit, index)
    unitary_coefficient_first = circuit.unitary()

    assert np.allclose(unitary_select_first, unitary_coefficient_first)
