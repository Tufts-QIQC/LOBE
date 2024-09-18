import pytest
import numpy as np
import cirq
from src.lobe.multiplexed_rotations import get_decomposed_multiplexed_rotation_circuit
from copy import copy


def _get_explicit_multiplexed_rotation_circuit(angles, is_controlled):
    circuit = cirq.Circuit()
    number_of_index = max(int(np.ceil(np.log2(len(angles)))), 1)
    counter = 0
    if is_controlled:
        ctrl = cirq.LineQubit(0)
        counter = 1
    index_register = [cirq.LineQubit(counter + i) for i in range(number_of_index)]
    rotation_qubit = cirq.LineQubit(counter + number_of_index)

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


def _get_decomposed_multiplexed_rotation_circuit(
    angles, is_controlled, with_ancilla=False
):
    number_of_index = max(int(np.ceil(np.log2(len(angles)))), 1)
    circuit = cirq.Circuit()
    ctrls = ([], [])
    clean_ancillae = []
    counter = 0
    if is_controlled:
        ctrls = ([cirq.LineQubit(0)], [1])
        counter = 1
        if with_ancilla:
            clean_ancillae = [cirq.LineQubit(0)]
            ctrls = ([cirq.LineQubit(1)], [1])
            counter = 2
    register = [cirq.LineQubit(i + counter) for i in range(number_of_index + 1)]

    circuit.append(
        get_decomposed_multiplexed_rotation_circuit(
            register, angles, clean_ancillae=clean_ancillae, ctrls=ctrls
        )
    )

    return circuit


@pytest.mark.parametrize("with_ancilla", [False, True])
@pytest.mark.parametrize("is_controlled", [False, True])
@pytest.mark.parametrize(
    "number_of_angles", [1] + np.random.randint(1, 1 << 8, size=20).tolist()
)
def test_multiplexed_rotation_circuit(number_of_angles, is_controlled, with_ancilla):
    angles = np.random.uniform(-1, 1, size=number_of_angles)
    undecomposed_circuit = _get_explicit_multiplexed_rotation_circuit(
        angles, is_controlled
    )
    decomposed_circuit = _get_decomposed_multiplexed_rotation_circuit(
        angles, is_controlled, with_ancilla=with_ancilla
    )

    if with_ancilla and is_controlled:
        num_non_ancilla_qubits = len(decomposed_circuit.all_qubits()) - 1
        assert np.allclose(
            undecomposed_circuit.unitary(),
            decomposed_circuit.unitary()[
                : 1 << num_non_ancilla_qubits, : 1 << num_non_ancilla_qubits
            ],
        )
    else:
        assert np.allclose(undecomposed_circuit.unitary(), decomposed_circuit.unitary())
