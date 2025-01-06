from .metrics import CircuitMetrics
import cirq


def decompose_controls_left(ctrls, clean_ancilla):
    _gates = []
    _metrics = CircuitMetrics()

    _metrics.add_to_clean_ancillae_usage(len(ctrls[0]) - 1)
    _metrics.number_of_elbows += len(ctrls[0]) - 1
    _gates.append(
        cirq.X.on(clean_ancilla).controlled_by(*ctrls[0], control_values=ctrls[1])
    )

    return _gates, _metrics


def decompose_controls_right(ctrls, clean_ancilla):
    _gates = []
    _metrics = CircuitMetrics()

    _gates.append(
        cirq.X.on(clean_ancilla).controlled_by(*ctrls[0], control_values=ctrls[1])
    )
    _metrics.add_to_clean_ancillae_usage(-len(ctrls[0]) + 1)

    return _gates, _metrics
