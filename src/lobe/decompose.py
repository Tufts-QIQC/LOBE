import cirq
from .metrics import CircuitMetrics


def decompose_controls_left(ctrls, clean_ancilla):
    """Decompose the controls representing a left-elbow onto a clean ancilla

    Args:
        - ctrls (Tuple(List[cirq.LineQubit], List[int])): A set of qubits and integers that correspond to
            the control qubits and values.
        - clean_ancilla (cirq.LineQubit): The clean ancilla which will store the logical AND of the controls

    Returns:
        - List of cirq operations representing the gates to be applied in the circuit
        - CircuitMetrics object representing cost of block-encoding circuit
    """
    _gates = []
    _metrics = CircuitMetrics()

    _metrics.add_to_clean_ancillae_usage(len(ctrls[0]) - 1)
    _metrics.number_of_elbows += len(ctrls[0]) - 1
    _gates.append(
        cirq.X.on(clean_ancilla).controlled_by(*ctrls[0], control_values=ctrls[1])
    )

    return _gates, _metrics


def decompose_controls_right(ctrls, clean_ancilla):
    """Decompose the controls representing a right-elbow freeing a clean ancilla

    Args:
        - ctrls (Tuple(List[cirq.LineQubit], List[int])): A set of qubits and integers that correspond to
            the control qubits and values.
        - clean_ancilla (cirq.LineQubit): The clean ancilla which will store the logical AND of the controls

    Returns:
        - List of cirq operations representing the gates to be applied in the circuit
        - CircuitMetrics object representing cost of block-encoding circuit
    """
    _gates = []
    _metrics = CircuitMetrics()

    _gates.append(
        cirq.X.on(clean_ancilla).controlled_by(*ctrls[0], control_values=ctrls[1])
    )
    _metrics.add_to_clean_ancillae_usage(-len(ctrls[0]) + 1)

    return _gates, _metrics
