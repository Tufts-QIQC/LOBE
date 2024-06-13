import pytest
import cirq
import numpy as np
from src.lobe.usp import add_naive_usp
from src.lobe.coefficient_oracle import add_coefficient_oracle
from src.lobe.select_oracle import add_select_oracle


@pytest.mark.parametrize(
    ["coefficients", "hamiltonian"],
    [
        (
            [1, 1, 1, 1],
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
            [1, 0.5, 0.5, 1],
            np.array(
                [
                    np.array([0, 0, 0, 0]),
                    np.array([0, 1, 0.5, 0]),
                    np.array([0, 0.5, 1, 0]),
                    np.array([0, 0, 0, 2]),
                ]
            ),
        ),
    ],
)
def test_block_encoding_for_toy_hamiltonian(coefficients, hamiltonian):
    operators = [(0, 0), (0, 1), (1, 0), (1, 1)]
    circuit = cirq.Circuit()
    validation = cirq.LineQubit(0)
    control = cirq.LineQubit(1)
    rotation = cirq.LineQubit(2)
    index = [cirq.LineQubit(i + 3) for i in range(2)]
    system = [cirq.LineQubit(i + 5) for i in range(2)]

    circuit = add_naive_usp(circuit, index)
    circuit.append(cirq.X.on(validation))
    circuit = add_select_oracle(circuit, validation, control, index, system, operators)
    circuit = add_coefficient_oracle(
        circuit, rotation, index, coefficients, len(operators)
    )
    circuit = add_naive_usp(circuit, index)

    upper_left_block = circuit.unitary()[: 1 << len(system), : 1 << len(system)]
    normalized_hamiltonian = hamiltonian / len(operators)
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
