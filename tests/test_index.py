import pytest
import cirq
import numpy as np
from src.lobe.index import index_over_terms
from src.lobe.metrics import CircuitMetrics
from src.lobe.system import System
from _utils import _validate_clean_ancillae_are_cleaned


@pytest.mark.parametrize("number_of_terms", np.random.randint((1 << 10) + 1, size=20))
def test_index_over_terms_metrics(number_of_terms):
    def _block_encoding_func(ctrls=([], [])):
        return [], CircuitMetrics()

    index_register = [
        cirq.LineQubit(i) for i in range(int(np.ceil(np.log2(number_of_terms))))
    ]
    clean_ancillae = [cirq.LineQubit(i + 100) for i in range(100)]
    ctrls = ([cirq.LineQubit(1000)], [1])

    block_encoding_functions = [_block_encoding_func] * number_of_terms

    _, metrics = index_over_terms(
        index_register,
        block_encoding_functions,
        clean_ancillae=clean_ancillae,
        ctrls=ctrls,
    )

    assert metrics.ancillae_highwater() == len(index_register)
    assert metrics.number_of_elbows == number_of_terms - 1


def test_index_over_terms_adds_metrics():
    def _block_encoding_func(ctrls=([], [])):
        metrics = CircuitMetrics()
        metrics.number_of_elbows += 10
        for _ in range(20):
            metrics.rotation_angles.append(0.0001)
        metrics.number_of_t_gates += 30
        metrics.add_to_clean_ancillae_usage(40)
        metrics.add_to_clean_ancillae_usage(-40)
        return [], metrics

    number_of_terms = 10
    index_register = [
        cirq.LineQubit(i) for i in range(int(np.ceil(np.log2(number_of_terms))))
    ]
    clean_ancillae = [cirq.LineQubit(i + 100) for i in range(100)]
    ctrls = ([cirq.LineQubit(1000)], [1])

    block_encoding_functions = [_block_encoding_func] * number_of_terms

    _, metrics = index_over_terms(
        index_register,
        block_encoding_functions,
        clean_ancillae=clean_ancillae,
        ctrls=ctrls,
    )

    assert metrics.number_of_elbows == number_of_terms - 1 + (number_of_terms * 10)
    assert metrics.number_of_nonclifford_rotations == (number_of_terms * 20)
    assert metrics.number_of_t_gates == (number_of_terms * 30)
    assert metrics.ancillae_highwater() == len(index_register) + 40


def test_index_lcu():
    ctrls = ([cirq.LineQubit(0)], [1])
    index_register = [cirq.LineQubit(i + 100) for i in range(int(np.ceil(np.log2(3))))]
    clean_ancillae = [cirq.LineQubit(i + 200) for i in range(100)]
    system = System(1, 300, number_of_fermionic_modes=1)

    def _apply_X(ctrls=([], [])):
        _gates = [
            cirq.X.on(system.fermionic_modes[0]).controlled_by(
                *ctrls[0], control_values=ctrls[1]
            )
        ]
        return _gates, CircuitMetrics()

    def _apply_Y(ctrls=([], [])):
        _gates = [
            cirq.Y.on(system.fermionic_modes[0]).controlled_by(
                *ctrls[0], control_values=ctrls[1]
            )
        ]
        return _gates, CircuitMetrics()

    def _apply_Z(ctrls=([], [])):
        _gates = [
            cirq.Z.on(system.fermionic_modes[0]).controlled_by(
                *ctrls[0], control_values=ctrls[1]
            )
        ]
        return _gates, CircuitMetrics()

    circuit = cirq.Circuit()
    circuit.append(cirq.X.on(ctrls[0][0]))
    circuit.append(cirq.H.on_each(*index_register))
    gates, _ = index_over_terms(
        index_register,
        [_apply_X, _apply_Y, _apply_Z],
        clean_ancillae=clean_ancillae,
        ctrls=ctrls,
    )
    circuit += gates
    circuit.append(cirq.H.on_each(*index_register))
    circuit.append(cirq.X.on(ctrls[0][0]))

    expected_block_encoding = (1 / 4) * np.asarray(
        [[0, 1], [1, 0]], dtype=np.complex128
    )
    expected_block_encoding += (1 / 4) * np.asarray([[0, -1j], [1j, 0]])
    expected_block_encoding += (1 / 4) * np.asarray([[1, 0], [0, -1]])
    expected_block_encoding += (1 / 4) * np.asarray([[1, 0], [0, 1]])

    upper_left_block = circuit.unitary()[:2, :2]

    assert np.allclose(upper_left_block, expected_block_encoding)

    _validate_clean_ancillae_are_cleaned(circuit, system, 2)


def test_index_controlled():
    ctrls = ([cirq.LineQubit(0)], [1])
    index_register = [cirq.LineQubit(1)]
    clean_ancillae = [cirq.LineQubit(i + 200) for i in range(100)]
    system = System(1, 300, number_of_fermionic_modes=1)

    def _apply_X(ctrls=([], [])):
        _gates = [
            cirq.X.on(system.fermionic_modes[0]).controlled_by(
                *ctrls[0], control_values=ctrls[1]
            )
        ]
        return _gates, CircuitMetrics()

    circuit = cirq.Circuit()
    circuit.append(cirq.H.on_each(*index_register))
    gates, _ = index_over_terms(
        index_register,
        [_apply_X],
        clean_ancillae=clean_ancillae,
        ctrls=ctrls,
    )
    circuit += gates
    circuit.append(cirq.H.on_each(*index_register))

    expected_block_encoding = np.eye(2)
    upper_left_block = circuit.unitary()[:2, :2]

    assert np.allclose(upper_left_block, expected_block_encoding)


def test_index_no_controls():
    index_register = [cirq.LineQubit(1)]
    clean_ancillae = [cirq.LineQubit(i + 200) for i in range(100)]
    system = System(1, 300, number_of_fermionic_modes=1)

    def _apply_X(ctrls=([], [])):
        _gates = [
            cirq.X.on(system.fermionic_modes[0]).controlled_by(
                *ctrls[0], control_values=ctrls[1]
            )
        ]
        return _gates, CircuitMetrics()

    circuit = cirq.Circuit()
    circuit.append(cirq.H.on_each(*index_register))
    gates, _ = index_over_terms(
        index_register,
        [_apply_X],
        clean_ancillae=clean_ancillae,
    )
    circuit += gates
    circuit.append(cirq.H.on_each(*index_register))

    expected_block_encoding = (1 / 2) * np.asarray([[1, 1], [1, 1]])
    upper_left_block = circuit.unitary()[:2, :2]

    assert np.allclose(upper_left_block, expected_block_encoding)
