def bosonic_mode_block_encoding(
    system,
    block_encoding_ancilla,
    active_index,
    exponents,
    clean_ancillae=[],
    ctrls=([], []),
):
    """Obtain operations for a block-encoding circuit for a product of bosonic ladder operators acting on the same mode

    NOTE: Assumes operator is written in the form: $(a_i^\dagger)^R (a_i)^S)$

    Args:
        system (lobe.system.System): The system object holding the system registers
        block_encoding_ancilla (cirq.LineQubit): The single block-encoding ancilla qubit
        active_index (int): An integer representing the bosonic mode on which the operator acts
        exponents (Tuple(int, int)): A Tuple of ints corresponds to the exponents of the operators: (S, R).
        clean_ancillae (List[cirq.LineQubit]): A list of qubits that are promised to start and end in the 0-state.
        ctrls (Tuple(List[cirq.LineQubit], List[int])): A set of qubits and integers that correspond to
            the control qubits and values.

    Returns:
        - List of cirq operations representing the gates to be applied in the circuit
        - CircuitMetrics object representing cost of block-encoding circuit
    """
    pass


def bosonic_mode_plus_hc_block_encoding(
    system,
    block_encoding_ancilla,
    active_index,
    exponents,
    clean_ancillae=[],
    ctrls=([], []),
):
    """Obtain operations for a block-encoding circuit for a product of bosonic ladder ops plus hermitian conjugate

    NOTE: Input arguements should correspond to only one term of the form: $(a_i^\dagger)^R (a_i)^S)$.
        The form of the hermitian conjugate will be inferred.

    Args:
        system (lobe.system.System): The system object holding the system registers
        block_encoding_ancilla (cirq.LineQubit): The single block-encoding ancilla qubit
        active_index (int): An integer representing the bosonic mode on which the operator acts
        exponents (Tuple(int, int)): A Tuple of ints corresponds to the exponents of the operators: (S, R).
        clean_ancillae (List[cirq.LineQubit]): A list of qubits that are promised to start and end in the 0-state.
        ctrls (Tuple(List[cirq.LineQubit], List[int])): A set of qubits and integers that correspond to
            the control qubits and values.

    Returns:
        - List of cirq operations representing the gates to be applied in the circuit
        - CircuitMetrics object representing cost of block-encoding circuit
    """
    pass
