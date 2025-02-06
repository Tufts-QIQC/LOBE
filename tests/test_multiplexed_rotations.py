import pytest
import numpy as np
import cirq
from src.lobe.multiplexed_rotations import (
    get_decomposed_multiplexed_rotation_circuit,
)
from src.lobe.system import System
from copy import copy
from _utils import _validate_clean_ancillae_are_cleaned


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
            cirq.ry(angles[index])
            .on(rotation_qubit)
            .controlled_by(
                *control_qubits,
                control_values=index_register_control_values,
            )
        )
    return circuit


@pytest.mark.parametrize(
    "number_of_angles",
    [1] + np.random.randint(1, 1 << 4, size=10).tolist() + [1 << 5],
)
def test_multiplexed_rotation_circuit_uncontrolled(number_of_angles):
    angles = np.random.uniform(-1, 1, size=number_of_angles)
    undecomposed_circuit = _get_explicit_multiplexed_rotation_circuit(angles, False)

    clean_ancillae = [cirq.LineQubit(i + 1) for i in range(100)]
    index_register = [
        cirq.LineQubit(i + 1000)
        for i in range(max(int(np.ceil(np.log2(number_of_angles))), 1))
    ]
    rotation_qubit = cirq.LineQubit(20000)

    system = System()
    system.number_of_system_qubits = len(index_register) + 1

    decomposed_gates, metrics = get_decomposed_multiplexed_rotation_circuit(
        index_register,
        rotation_qubit,
        angles,
        clean_ancillae=clean_ancillae,
    )
    decomposed_circuit = cirq.Circuit(decomposed_gates)

    assert np.allclose(
        undecomposed_circuit.unitary(),
        decomposed_circuit.unitary()[
            : 1 << (len(index_register) + 1), : 1 << (len(index_register) + 1)
        ],
    )
    assert metrics.ancillae_highwater() == 0
    assert metrics.number_of_elbows == 0


@pytest.mark.parametrize(
    "number_of_angles",
    [1] + np.random.randint(1, 1 << 4, size=10).tolist() + [1 << 5],
)
def test_get_decomposed_multiplexed_rotation_circuit_controlled(number_of_angles):
    angles = np.random.uniform(-1, 1, size=number_of_angles)
    ctrls = ([cirq.LineQubit(0)], [1])
    clean_ancillae = [cirq.LineQubit(-i - 1) for i in range(100)]
    index_register = [
        cirq.LineQubit(i + 1000)
        for i in range(max(int(np.ceil(np.log2(number_of_angles))), 1))
    ]
    rotation_qubit = cirq.LineQubit(20000)

    system = System()
    system.number_of_system_qubits = len(index_register) + 1

    expected_circuit = _get_explicit_multiplexed_rotation_circuit(angles, True)

    gates, metrics = get_decomposed_multiplexed_rotation_circuit(
        index_register,
        rotation_qubit,
        angles,
        clean_ancillae=clean_ancillae,
        ctrls=ctrls,
    )
    circuit = cirq.Circuit(gates)

    _validate_clean_ancillae_are_cleaned(circuit, system, 0)
    assert np.allclose(
        circuit.unitary()[
            : 1 << (len(index_register) + 2), : 1 << (len(index_register) + 2)
        ],
        expected_circuit.unitary(),
    )
    assert metrics.clean_ancillae_usage[-1] == 0
    assert metrics.ancillae_highwater() == len(index_register)
    assert metrics.number_of_elbows == len(index_register)
    assert metrics.number_of_nonclifford_rotations <= (1 << len(index_register)) + 2


def test_get_decomposed_multiplexed_rotation_circuit_only_counts_nonclifford_rotations():
    angles = [
        0,
        0.5,
        0.125,
        0.375,
    ]  # should result in one rotation by 0 rads and three nonClifford rotations

    number_of_index = max(int(np.ceil(np.log2(len(angles)))), 1)
    ctrls = ([], [])
    clean_ancillae = [cirq.LineQubit(-1 - i) for i in range(100)]
    counter = 0
    register = [cirq.LineQubit(i + counter) for i in range(number_of_index + 1)]

    _, metrics = get_decomposed_multiplexed_rotation_circuit(
        register[:-1],
        register[-1],
        angles,
        clean_ancillae=clean_ancillae,
        ctrls=ctrls,
    )

    assert metrics.number_of_nonclifford_rotations == 3


@pytest.mark.parametrize(
    "number_of_angles",
    [1] + np.random.randint(1, 1 << 4, size=10).tolist() + [1 << 5],
)
def test_multiplexed_rotation_circuit_daggered(number_of_angles):
    angles = np.random.uniform(-1, 1, size=number_of_angles)
    undecomposed_circuit = _get_explicit_multiplexed_rotation_circuit(angles, False)

    clean_ancillae = [cirq.LineQubit(i + 1) for i in range(100)]
    index_register = [
        cirq.LineQubit(i + 1000)
        for i in range(max(int(np.ceil(np.log2(number_of_angles))), 1))
    ]
    rotation_qubit = cirq.LineQubit(20000)

    system = System()
    system.number_of_system_qubits = len(index_register) + 1

    decomposed_gates, _ = get_decomposed_multiplexed_rotation_circuit(
        index_register,
        rotation_qubit,
        angles,
        clean_ancillae=clean_ancillae,
    )
    decomposed_circuit = cirq.Circuit(decomposed_gates)

    assert np.allclose(
        undecomposed_circuit.unitary(),
        decomposed_circuit.unitary()[
            : 1 << (len(index_register) + 1), : 1 << (len(index_register) + 1)
        ],
    )

    decomposed_gates, _ = get_decomposed_multiplexed_rotation_circuit(
        index_register,
        rotation_qubit,
        angles,
        dagger=True,
        clean_ancillae=clean_ancillae,
    )
    decomposed_circuit.append(decomposed_gates)

    assert np.allclose(
        np.eye(1 << len(decomposed_circuit.all_qubits())),
        decomposed_circuit.unitary(),
    )
