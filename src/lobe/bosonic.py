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

    gates = []
    block_encoding_metrics = CircuitMetrics()

    # gates.append(cirq.H.on(block_encoding_ancilla))
    R, S = exponents[0], exponents[1]

    adder_controls = (ctrls[0] + [block_encoding_ancilla], ctrls[1] + [0])
    gates += add_classical_value_gate_efficient(
        system.bosonic_system[active_index],
        R - S,
        clean_ancillae=clean_ancillae,
        ctrls=adder_controls,
    )

    gates += _add_multi_bosonic_rotations(
        block_encoding_ancilla,
        system.bosonic_system[active_index],
        R,
        S,
        clean_ancillae=clean_ancillae,
        ctrls=ctrls,
    )

    # gates.append(cirq.H.on(block_encoding_ancilla))

    return gates, block_encoding_metrics


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


def _get_bosonic_rotation_angles(
    maximum_occupation_number, creation_exponent, annhilation_exponent
):
    """Get the associated Ry rotation angles for an operator of the form: $a_i^\dagger^R a_i^S$

    Args:
        maximum_occupation_number (int): The maximum allowed bosonic occupation ($\Omega$)
        creation_exponent (int): The exponent on the creation operator (R)
        annihilation_exponent (int): The exponent on the annihilation operator (R)

    Returns:
        - List of floats
    """
    intended_coefficients = [1] * (maximum_occupation_number + 1)
    for omega in range(maximum_occupation_number + 1):
        if (omega - annhilation_exponent) < 0:
            intended_coefficients[omega] = 0
        elif (
            omega - annhilation_exponent + creation_exponent
        ) > maximum_occupation_number:
            intended_coefficients[omega] = 0
        else:
            for r in range(creation_exponent):
                intended_coefficients[omega] *= np.sqrt(
                    (omega - r) / maximum_occupation_number
                )
            for s in range(1, annhilation_exponent + 1):
                intended_coefficients[omega] *= np.sqrt(
                    (omega - creation_exponent + s - 1) / maximum_occupation_number
                )

    rotation_angles = [
        2 / np.pi * np.arccos(intended_coefficient)
        for intended_coefficient in intended_coefficients
    ]
    return rotation_angles
