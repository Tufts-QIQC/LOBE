import pytest
import numpy as np
import cirq
from src.lobe.multiplexed_rotations import get_decomposed_multiplexed_rotation_circuit


def _get_explicit_multiplexed_rotation_circuit(angles):
    circuit = cirq.Circuit()
    number_of_index = int(np.ceil(np.log2(len(angles))))
    index_register = [cirq.LineQubit(i) for i in range(number_of_index)]
    rotation_qubit = cirq.LineQubit(number_of_index)

    for index in range(len(angles)):
        index_register_control_values = [
            int(i) for i in format(index, f"#0{2+number_of_index}b")[2:]
        ]
        circuit.append(
            cirq.ry(np.pi * angles[index])
            .on(rotation_qubit)
            .controlled_by(
                *index_register, control_values=index_register_control_values
            )
        )
    return circuit


def _get_decomposed_multiplexed_rotation_circuit(angles):
    number_of_index = int(np.ceil(np.log2(len(angles))))
    circuit = cirq.Circuit()
    register = [cirq.LineQubit(i) for i in range(number_of_index + 1)]

    circuit.append(get_decomposed_multiplexed_rotation_circuit(register, angles))

    return circuit


@pytest.mark.parametrize("number_of_angles", np.random.randint(0, 1 << 8, size=20))
def test_multiplexed_rotation_circuit(number_of_angles):
    angles = np.random.uniform(-1, 1, size=number_of_angles)
    undecomposed_circuit = _get_explicit_multiplexed_rotation_circuit(angles)
    decomposed_circuit = _get_decomposed_multiplexed_rotation_circuit(angles)
    assert np.allclose(undecomposed_circuit.unitary(), decomposed_circuit.unitary())
