from src.lobe.metrics import CircuitMetrics
from src.lobe.addition import add_classical_value
import numpy as np
import cirq
from .decompose import decompose_controls_left, decompose_controls_right
from src.lobe.multiplexed_rotations import get_decomposed_multiplexed_rotation_circuit


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
        exponents (Tuple(int, int)): A Tuple of ints corresponds to the exponents of the operators: (R, S).
        clean_ancillae (List[cirq.LineQubit]): A list of qubits that are promised to start and end in the 0-state.
        ctrls (Tuple(List[cirq.LineQubit], List[int])): A set of qubits and integers that correspond to
            the control qubits and values.

    Returns:
        - List of cirq operations representing the gates to be applied in the circuit
        - CircuitMetrics object representing cost of block-encoding circuit
    """

    gates = []
    block_encoding_metrics = CircuitMetrics()

    R, S = exponents[0], exponents[1]

    adder_gates, adder_metrics = add_classical_value(
        system.bosonic_system[active_index],
        R - S,
        clean_ancillae=clean_ancillae,
        ctrls=ctrls,
    )
    gates += adder_gates
    block_encoding_metrics += adder_metrics

    rotation_gates, rotation_metrics = _add_multi_bosonic_rotations(
        block_encoding_ancilla,
        system.bosonic_system[active_index],
        R,
        S,
        clean_ancillae=clean_ancillae,
        ctrls=ctrls,
    )
    gates += rotation_gates
    block_encoding_metrics += rotation_metrics

    return gates, block_encoding_metrics


def bosonic_modes_block_encoding(
    system,
    block_encoding_ancillae,
    active_indices,
    exponents_list,
    clean_ancillae=[],
    ctrls=([], []),
):
    """Obtain operations for a block-encoding circuit for a product of bosonic operators acting on multiple modes

    NOTE: Assumes operator is written in the form:
        $(a_i^\dagger)^{R_i} (a_i)^{S_i}) (a_j^\dagger)^{R_j} (a_j)^{S_j}) ... (a_l^\dagger)^{R_l} (a_l)^{S_l})$

    Args:
        system (lobe.system.System): The system object holding the system registers
        block_encoding_ancillae (List[cirq.LineQubit]): A list of ancillae with length matching the number of
            active indices. Each block-encoding ancilla is used to block-encode the operators acting on one mode
        active_indices (List[int]): An integer representing the bosonic modes on which the operator acts in
            right to left order: [l, ..., j, i]
        exponents_list (List[Tuple(int, int)]): A List of Tuples of ints corresponds to the exponents of the operators: (R, S).
        clean_ancillae (List[cirq.LineQubit]): A list of qubits that are promised to start and end in the 0-state.
        ctrls (Tuple(List[cirq.LineQubit], List[int])): A set of qubits and integers that correspond to
            the control qubits and values.

    Returns:
        - List of cirq operations representing the gates to be applied in the circuit
        - CircuitMetrics object representing cost of block-encoding circuit
    """

    gates = []
    block_encoding_metrics = CircuitMetrics()
    for block_encoding_ancilla, active_index, exponents in zip(
        block_encoding_ancillae, active_indices, exponents_list
    ):
        _gates, _metrics = bosonic_mode_block_encoding(
            system,
            block_encoding_ancilla,
            active_index,
            exponents,
            clean_ancillae=clean_ancillae,
            ctrls=ctrls,
        )
        gates += _gates
        block_encoding_metrics += _metrics
    return gates, block_encoding_metrics


def bosonic_mode_plus_hc_block_encoding(
    system,
    block_encoding_ancillae,
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
    gates = []
    block_encoding_metrics = CircuitMetrics()
    index = block_encoding_ancillae[0]

    gates.append(cirq.H.on(index))

    block_encoding_metrics.add_to_clean_ancillae_usage(1)
    _gates, _metrics = decompose_controls_left(
        (ctrls[0] + [index], ctrls[1] + [0]), clean_ancillae[0]
    )
    gates += _gates
    block_encoding_metrics += _metrics

    adder_gates, adder_metrics = add_classical_value(
        system.bosonic_system[active_index],
        exponents[0] - exponents[1],
        clean_ancillae=clean_ancillae[1:],
        ctrls=([clean_ancillae[0]], [1]),
    )
    gates += adder_gates
    block_encoding_metrics += adder_metrics

    rotation_gates, rotation_metrics = _add_multi_bosonic_rotations(
        block_encoding_ancillae[1],
        system.bosonic_system[active_index],
        exponents[0],
        exponents[1],
        clean_ancillae=clean_ancillae[1:],
        ctrls=ctrls,
    )
    gates += rotation_gates
    block_encoding_metrics += rotation_metrics

    gates.append(
        cirq.X.on(clean_ancillae[0]).controlled_by(*ctrls[0], control_values=ctrls[1])
    )
    adder_gates, adder_metrics = add_classical_value(
        system.bosonic_system[active_index],
        -exponents[0] + exponents[1],
        clean_ancillae=clean_ancillae[1:],
        ctrls=([clean_ancillae[0]], [1]),
    )
    gates += adder_gates
    block_encoding_metrics += adder_metrics

    _gates, _metrics = decompose_controls_right(
        (ctrls[0] + [index], ctrls[1] + [1]), clean_ancillae[0]
    )
    gates += _gates
    block_encoding_metrics += _metrics
    block_encoding_metrics.add_to_clean_ancillae_usage(-1)

    gates.append(cirq.H.on(index))

    return gates, block_encoding_metrics


def bosonic_product_plus_hc_block_encoding(
    system,
    block_encoding_ancillae,
    active_indices,
    exponents,
    clean_ancillae=[],
    ctrls=([], []),
):
    """
    NOTE: Assumes operator is written in the form:
         $(a_i^\dagger)^{R_i} (a_i)^{S_i}) (a_j^\dagger)^{R_j} (a_j)^{S_j}) ... (a_l^\dagger)^{R_l} (a_l)^{S_l})$
         Hermitian conjugate

     Args:
         system (lobe.system.System): The system object holding the system registers
         block_encoding_ancillae (List[cirq.LineQubit]): The block-encoding ancillae qubits
         active_indices (List[int]): A list of the modes upon which the ladder operators act. Assumed to be in order
             of which operators are applied (right to left).
         exponents (List[tuple]): A list of tuples (Ri, Si) containing the number of creation (Ri) and
             annihilation (Si) operators in the operator acting on mode i.
         clean_ancillae (List[cirq.LineQubit]): A list of qubits that are promised to start and end in the 0-state.
         ctrls (Tuple(List[cirq.LineQubit], List[int])): A set of qubits and integers that correspond to
             the control qubits and values.

     Returns:
         - List of cirq operations representing the gates to be applied in the circuit
         - CircuitMetrics object representing cost of block-encoding circuit
    """

    assert len(ctrls[0]) <= 1
    if len(ctrls[0]) == 1:
        assert ctrls[1] == [1]

    gates = []
    block_encoding_metrics = CircuitMetrics()
    index = block_encoding_ancillae[0]

    gates.append(cirq.H.on(index))

    block_encoding_metrics.add_to_clean_ancillae_usage(1)
    _gates, _metrics = decompose_controls_left(
        (ctrls[0] + [index], ctrls[1] + [0]), clean_ancillae[0]
    )
    gates += _gates
    block_encoding_metrics += _metrics

    for active_index in active_indices:
        Ri, Si = exponents[active_index][0], exponents[active_index][1]

        adder_gates, adder_metrics = add_classical_value(
            system.bosonic_system[active_index],
            Ri - Si,
            clean_ancillae=clean_ancillae[1:],
            ctrls=([clean_ancillae[0]], [1]),
        )
        gates += adder_gates
        block_encoding_metrics += adder_metrics

        rotation_gates, rotation_metrics = _add_multi_bosonic_rotations(
            block_encoding_ancillae[1],
            system.bosonic_system[active_index],
            Ri,
            Si,
            clean_ancillae=clean_ancillae[1:],
            ctrls=ctrls,
        )
        gates += rotation_gates
        block_encoding_metrics += rotation_metrics

        gates.append(
            cirq.X.on(clean_ancillae[0]).controlled_by(
                *ctrls[0], control_values=ctrls[1]
            )
        )  # right elbow followed by left elbow is a CNOT
        adder_gates, adder_metrics = add_classical_value(
            system.bosonic_system[active_index],
            -Ri + Si,
            clean_ancillae=clean_ancillae[1:],
            ctrls=([clean_ancillae[0]], [1]),
        )
        gates += adder_gates
        block_encoding_metrics += adder_metrics

        _gates, _metrics = decompose_controls_right(
            (ctrls[0] + [index], ctrls[1] + [1]), clean_ancillae[0]
        )
        gates += _gates
        block_encoding_metrics += _metrics
        block_encoding_metrics.add_to_clean_ancillae_usage(-1)

        gates.append(cirq.H.on(index))

    return gates, block_encoding_metrics


def _add_multi_bosonic_rotations(
    rotation_qubit,
    bosonic_mode_register,
    creation_exponent=0,
    annihilation_exponent=0,
    clean_ancillae=[],
    ctrls=([], []),
):
    """Add rotations to pickup bosonic coefficients corresponding to a series of ladder operators (assumed
        to be normal ordered) acting on one bosonic mode within a term.

    Args:
        rotation_qubit (cirq.LineQubit): The qubit that is rotated to pickup the amplitude corresponding
            to the coefficients that appear when a bosonic op hits a quantum state
        bosonic_mode_register (List[cirq.LineQubit]): The qubits that store the occupation of the bosonic
            mode being acted upon.
        creation_exponent (int): The number of subsequent creation operators in the term
        annihilation_exponent (int): The number of subsequent annihilation operators in the term
        clean_ancillae (List[cirq.LineQubit]): A list of qubits that are promised to start and end in the 0-state.
        ctrls (Tuple(List[cirq.LineQubit], List[int])): A set of qubits and integers that correspond to
            the control qubits and values.

    Returns:
        - The gates to perform the unitary operation
    """
    maximum_occupation_number = (1 << len(bosonic_mode_register)) - 1

    angles = _get_bosonic_rotation_angles(
        maximum_occupation_number, creation_exponent, annihilation_exponent
    )

    return get_decomposed_multiplexed_rotation_circuit(
        bosonic_mode_register,
        rotation_qubit,
        angles,
        clean_ancillae=clean_ancillae,
        ctrls=ctrls,
    )


def _get_bosonic_rotation_angles(
    maximum_occupation_number, creation_exponent, annihilation_exponent
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
        if (omega - creation_exponent) < 0:
            intended_coefficients[omega] = 0
        elif (
            omega - creation_exponent + annihilation_exponent
        ) > maximum_occupation_number:
            intended_coefficients[omega] = 0
        else:
            for r in range(creation_exponent):
                intended_coefficients[omega] *= np.sqrt(
                    (omega - r) / maximum_occupation_number
                )
            for s in range(annihilation_exponent):
                intended_coefficients[omega] *= np.sqrt(
                    (omega - creation_exponent + s + 1) / maximum_occupation_number
                )

    rotation_angles = [
        2 / np.pi * np.arccos(intended_coefficient)
        for intended_coefficient in intended_coefficients
    ]
    return rotation_angles
