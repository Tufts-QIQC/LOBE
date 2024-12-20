def phi4_interaction_term_block_encoding(
    system,
    block_encoding_ancillae,
    active_indices,
    clean_ancillae=[],
    ctrls=([], []),
):
    """Add a block-encoding circuit for the phi4 interaction term

    Args:
        system (lobe.system.System): The system object holding the system registers
        block_encoding_ancillae (List[cirq.LineQubit]): The block-encoding ancillae qubits
        active_indices (List[int]): A list of the modes upon which the ladder operators act. Assumed to be in order
            of which operators are applied (right to left).
        clean_ancillae (List[cirq.LineQubit]): A list of qubits that are promised to start and end in the 0-state.
        ctrls (Tuple(List[cirq.LineQubit], List[int])): A set of qubits and integers that correspond to
            the control qubits and values.

    Returns:
        - List of cirq operations representing the gates to be applied in the circuit
        - CircuitMetrics object representing cost of block-encoding circuit
    """
    pass


def yukawa_4point_pair_term_block_encoding(
    system,
    block_encoding_ancillae,
    active_indices,
    clean_ancillae=[],
    ctrls=([], []),
):
    """Add a block-encoding circuit for the 4 point pair term in the full Yukawa model

    Args:
        system (lobe.system.System): The system object holding the system registers
        block_encoding_ancillae (List[cirq.LineQubit]): The block-encoding ancillae qubits
        active_indices (List[int]): A list of the modes upon which the ladder operators act. Assumed to be in order
            of which operators are applied (right to left).
        clean_ancillae (List[cirq.LineQubit]): A list of qubits that are promised to start and end in the 0-state.
        ctrls (Tuple(List[cirq.LineQubit], List[int])): A set of qubits and integers that correspond to
            the control qubits and values.

    Returns:
        - List of cirq operations representing the gates to be applied in the circuit
        - CircuitMetrics object representing cost of block-encoding circuit
    """
    pass


def yukawa_3point_pair_term_block_encoding(
    system,
    block_encoding_ancillae,
    active_indices,
    clean_ancillae=[],
    ctrls=([], []),
):
    """Add a block-encoding circuit for the 3 point pair term in the full Yukawa model

    Args:
        system (lobe.system.System): The system object holding the system registers
        block_encoding_ancillae (List[cirq.LineQubit]): The block-encoding ancillae qubits
        active_indices (List[int]): A list of the modes upon which the ladder operators act. Assumed to be in order
            of which operators are applied (right to left).
        clean_ancillae (List[cirq.LineQubit]): A list of qubits that are promised to start and end in the 0-state.
        ctrls (Tuple(List[cirq.LineQubit], List[int])): A set of qubits and integers that correspond to
            the control qubits and values.

    Returns:
        - List of cirq operations representing the gates to be applied in the circuit
        - CircuitMetrics object representing cost of block-encoding circuit
    """
    pass


def _custom_fermionic_plus_nonhc_block_encoding(
    system,
    block_encoding_ancilla,
    active_indices,
    clean_ancillae=[],
    ctrls=([], []),
):
    """Add a block-encoding circuit for the operator: $b_i b_j b_k^\dagger + b_j^\dagger b_i^\dagger b_k^\dagger$.

    NOTE: Input arguements should correspond to only one term. The form of the hermitian conjugate will be inferred.

    Args:
        system (lobe.system.System): The system object holding the system registers
        block_encoding_ancilla (cirq.LineQubit): The single block-encoding ancilla qubit
        active_indices (List[int]): A list of the modes upon which the ladder operators act. Assumed to be in order
            of which operators are applied (right to left): [k, j, i].
        clean_ancillae (List[cirq.LineQubit]): A list of qubits that are promised to start and end in the 0-state.
        ctrls (Tuple(List[cirq.LineQubit], List[int])): A set of qubits and integers that correspond to
            the control qubits and values.

    Returns:
        - List of cirq operations representing the gates to be applied in the circuit
        - CircuitMetrics object representing cost of block-encoding circuit
    """
    pass


def _custom_term_block_encoding(
    system,
    block_encoding_ancilla,
    active_indices,
    clean_ancillae=[],
    ctrls=([], []),
):
    """Add a block-encoding circuit for the operator: $b_i a_j + b_i^\dagger a_j^\dagger$.

    NOTE: Input arguements should correspond to only one term. The form of the hermitian conjugate will be inferred.

    Args:
        system (lobe.system.System): The system object holding the system registers
        block_encoding_ancilla (cirq.LineQubit): The single block-encoding ancilla qubit
        active_indices (List[int]): A list of the modes upon which the ladder operators act. Assumed to be in order
            of which operators are applied (right to left): [j, i].
        clean_ancillae (List[cirq.LineQubit]): A list of qubits that are promised to start and end in the 0-state.
        ctrls (Tuple(List[cirq.LineQubit], List[int])): A set of qubits and integers that correspond to
            the control qubits and values.

    Returns:
        - List of cirq operations representing the gates to be applied in the circuit
        - CircuitMetrics object representing cost of block-encoding circuit
    """
    pass
