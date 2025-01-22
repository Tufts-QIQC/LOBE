from .metrics import CircuitMetrics
from .decompose import decompose_controls_left, decompose_controls_right
import numpy as np


def index_over_terms(
    index_register, block_encoding_functions, clean_ancillae, ctrls=([], [])
):
    """Create a block-encoding of a linear combination of block-encodings

    Args:
        index_register (List[cirq.LineQubit]): The qubit register that indexes the terms
        block_encoding_functions (List[Callable]): A list of functions that apply the block-encoding for each operator
            in the list when called. Function signature should only accept ctrls
        clean_ancillae (List[cirq.LineQubit]): A list of qubits that are promised to start and end in the 0-state.
        ctrls (Tuple(List[cirq.LineQubit], List[int])): A set of qubits and integers that correspond to
            the control qubits and values.

    Returns:
        - List of cirq operations representing the gates to be applied in the circuit
        - CircuitMetrics object representing cost of block-encoding circuit
    """
    assert len(ctrls[0]) <= 1
    if len(ctrls[0]) > 0:
        assert ctrls[1] == [1]
    gates = []
    block_encoding_metrics = CircuitMetrics()

    number_of_terms = len(block_encoding_functions)
    block_encoding_metrics.number_of_elbows += number_of_terms - 1
    block_encoding_metrics.add_to_clean_ancillae_usage(
        int(np.ceil(np.log2(number_of_terms)))
    )

    for index in range(number_of_terms):
        index_ancilla = clean_ancillae[0]
        # Get binary control values corresponding to index
        index_register_control_values = [
            int(i) for i in format(index, f"#0{2+len(index_register)}b")[2:]
        ]
        # Set index control
        _gates, _ = decompose_controls_left(
            (index_register + ctrls[0], index_register_control_values + ctrls[1]),
            index_ancilla,
        )
        gates += _gates

        # Apply block-encoding function
        _gates, _metrics = block_encoding_functions[index](ctrls=([index_ancilla], [1]))
        gates += _gates
        block_encoding_metrics += _metrics

        # Release index control
        _gates, _ = decompose_controls_right(
            (index_register + ctrls[0], index_register_control_values + ctrls[1]),
            index_ancilla,
        )
        gates += _gates

    block_encoding_metrics.add_to_clean_ancillae_usage(
        -int(np.ceil(np.log2(number_of_terms)))
    )
    return gates, block_encoding_metrics
