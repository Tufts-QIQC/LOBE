import pytest
import numpy as np
import cirq
from src.lobe.multiplexed_rotations import get_decomposed_multiplexed_rotation_circuit
from copy import copy


def _get_explicit_multiplexed_rotation_circuit(angles, is_controlled):
    circuit = cirq.Circuit()
    number_of_index = int(np.ceil(np.log2(len(angles))))
    index_register = [cirq.LineQubit(i) for i in range(number_of_index)]
    rotation_qubit = cirq.LineQubit(number_of_index)
    if is_controlled:
        ctrl = cirq.LineQubit(number_of_index + 1)

    for index in range(len(angles)):
        index_register_control_values = [
            int(i) for i in format(index, f"#0{2+number_of_index}b")[2:]
        ]
        control_qubits = copy(index_register)
        if is_controlled:
            control_qubits.append(ctrl)
            index_register_control_values.append(1)
        circuit.append(
            cirq.ry(np.pi * angles[index])
            .on(rotation_qubit)
            .controlled_by(
                *control_qubits,
                control_values=index_register_control_values,
            )
        )
    return circuit


def _get_decomposed_multiplexed_rotation_circuit(angles, is_controlled):
    number_of_index = int(np.ceil(np.log2(len(angles))))
    circuit = cirq.Circuit()
    register = [cirq.LineQubit(i) for i in range(number_of_index + 1)]
    ctrls = ([], [])
    if is_controlled:
        ctrls = ([cirq.LineQubit(number_of_index + 1)], [1])

    circuit.append(
        get_decomposed_multiplexed_rotation_circuit(register, angles, ctrls=ctrls)
    )

    return circuit


@pytest.mark.parametrize("is_controlled", [False, True])
@pytest.mark.parametrize("number_of_angles", np.random.randint(1, 1 << 8, size=20))
def test_multiplexed_rotation_circuit(number_of_angles, is_controlled):
    angles = np.random.uniform(-1, 1, size=number_of_angles)
    undecomposed_circuit = _get_explicit_multiplexed_rotation_circuit(
        angles, is_controlled
    )
    decomposed_circuit = _get_decomposed_multiplexed_rotation_circuit(
        angles, is_controlled
    )
    assert np.allclose(undecomposed_circuit.unitary(), decomposed_circuit.unitary())
