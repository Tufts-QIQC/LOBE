from src.lobe.numerical_comparator import is_less_than, is_greater_than
import pytest
import cirq
import numpy as np
from src.lobe._utils import pretty_print


@pytest.mark.parametrize(
    "number_of_qubits", [1, 2, 3, 4, 5] + np.random.randint(1, 10, size=5).tolist()
)
@pytest.mark.parametrize(
    "value",
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    + np.random.randint(0, 1 << 10, size=20).tolist(),
)
@pytest.mark.parametrize(
    "reference", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
)
def test_is_less_than(number_of_qubits, value, reference):
    value = value % (1 << number_of_qubits)
    reference = reference % (1 << number_of_qubits)
    if reference == 0:
        return

    number_of_ancillae = 4

    qbool = cirq.LineQubit(0)
    ancillae = [cirq.LineQubit(i + 1) for i in range(number_of_ancillae)]
    qubits = [
        cirq.LineQubit(i + 1 + number_of_ancillae) for i in range(number_of_qubits)
    ]
    circuit = cirq.Circuit()
    circuit.append(cirq.I.on_each(*ancillae))
    circuit.append(cirq.I.on_each(*qubits))

    gates, qbool, number_of_used_ancillae = is_less_than(
        qubits, reference, qbool, clean_ancillae=ancillae
    )
    circuit += gates

    initial_state = np.zeros(1 << (number_of_ancillae + number_of_qubits + 1))
    initial_state[value] = 1
    simulator = cirq.Simulator()

    output_wavefunction = simulator.simulate(
        circuit, initial_state=initial_state
    ).final_state_vector

    index = np.nonzero(output_wavefunction)[0][0]

    expected_state = (
        "0" * number_of_ancillae + format(value, f"#0{2+number_of_qubits}b")[2:]
    )

    if value < reference:
        expected_state = "1" + expected_state
    else:
        expected_state = "0" + expected_state

    assert len(expected_state) == number_of_qubits + number_of_ancillae + 1

    assert index == int(expected_state, 2)


@pytest.mark.parametrize(
    "number_of_qubits", [1, 2, 3, 4, 5] + np.random.randint(1, 10, size=5).tolist()
)
@pytest.mark.parametrize(
    "value",
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    + np.random.randint(0, 1 << 10, size=20).tolist(),
)
@pytest.mark.parametrize(
    "reference", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
)
def test_is_greater_than(number_of_qubits, value, reference):
    value = value % (1 << number_of_qubits)
    reference = reference % (1 << number_of_qubits)
    reference = (1 << number_of_qubits) - reference

    if not (reference < ((1 << number_of_qubits) - 1)):
        return
    if not (reference > (((1 << number_of_qubits) - 1) - 15)):
        return

    number_of_ancillae = 4

    qbool = cirq.LineQubit(0)
    ancillae = [cirq.LineQubit(i + 1) for i in range(number_of_ancillae)]
    qubits = [
        cirq.LineQubit(i + 1 + number_of_ancillae) for i in range(number_of_qubits)
    ]
    circuit = cirq.Circuit()
    circuit.append(cirq.I.on_each(*ancillae))
    circuit.append(cirq.I.on_each(*qubits))

    gates, qbool, number_of_used_ancillae = is_greater_than(
        qubits, reference, qbool, clean_ancillae=ancillae
    )
    circuit += gates

    initial_state = np.zeros(1 << (number_of_ancillae + number_of_qubits + 1))
    initial_state[value] = 1
    simulator = cirq.Simulator()

    output_wavefunction = simulator.simulate(
        circuit, initial_state=initial_state
    ).final_state_vector

    index = np.nonzero(output_wavefunction)[0][0]

    expected_state = (
        "0" * number_of_ancillae + format(value, f"#0{2+number_of_qubits}b")[2:]
    )

    if value > reference:
        expected_state = "1" + expected_state
    else:
        expected_state = "0" + expected_state

    assert len(expected_state) == number_of_qubits + number_of_ancillae + 1

    assert index == int(expected_state, 2)
