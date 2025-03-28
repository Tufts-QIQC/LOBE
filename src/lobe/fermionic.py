import cirq
import math
from .metrics import CircuitMetrics


def fermionic_product_block_encoding(
    system,
    block_encoding_ancillae,
    active_indices,
    operator_types,
    sign=1,
    clean_ancillae=[],
    ctrls=([], []),
):
    """Obtain operations for a block-encoding circuit for a product of fermionic ladder operators

    Args:
        - system (lobe.system.System): The system object holding the system registers
        - block_encoding_ancillae (List[cirq.LineQubit]): A list with a single block-encoding ancilla qubit
        - active_indices (List[int]): A list of the modes upon which the ladder operators act. Assumed to be in order
            of which operators are applied (right to left).
        - operator_types (List[int]): A list of ints indicating the type of ladder operators acting on each mode. Set
            to 0 if operator is a annihilation/lowering ladder operator (b_i). 1 if creation/raising ladder operator
            (b_i^\dagger). 2 if b_i^\dagger b_i. 3 if b_i b_i^dagger
        - sign (int): Either 1 or -1 to indicate the sign of the term
        - clean_ancillae (List[cirq.LineQubit]): A list of qubits that are promised to start and end in the 0-state.
        - ctrls (Tuple(List[cirq.LineQubit], List[int])): A set of qubits and integers that correspond to
            the control qubits and values.

    Returns:
        - List of cirq operations representing the gates to be applied in the circuit
        - CircuitMetrics object representing cost of block-encoding circuit
    """
    assert len(ctrls[0]) == 1
    assert ctrls[1] == [1]
    assert len(block_encoding_ancillae) == 1
    block_encoding_ancilla = block_encoding_ancillae[0]
    block_encoding_metrics = CircuitMetrics()
    gates = []

    if sign == -1:
        gates.append(cirq.Z.on(ctrls[0][0]))

    if len(active_indices) == 1:
        block_encoding_metrics.number_of_elbows += 1
        block_encoding_metrics.add_to_clean_ancillae_usage(1)
        gates.append(
            cirq.X.on(block_encoding_ancilla).controlled_by(
                *ctrls[0],
                system.fermionic_modes[active_indices[0]],
                control_values=ctrls[1] + [int(operator_types[0] % 2)],
            )
        )
        block_encoding_metrics.add_to_clean_ancillae_usage(-1)
        if (operator_types[0] != 2) and (operator_types[0] != 3):
            op_gates, op_metrics = _apply_fermionic_ladder_op(
                system, active_indices[0], ctrls=ctrls
            )
            gates += op_gates
            block_encoding_metrics += op_metrics

        return gates, block_encoding_metrics

    temporary_computations = []
    clean_ancillae_index = 0

    non_number_op_indices = []
    non_number_op_types = []
    number_op_qubits = []
    number_op_controls = []
    for type, index in zip(operator_types, active_indices):
        if (type != 2) and (type != 3):
            non_number_op_types.append(type)
            non_number_op_indices.append(index)
        else:
            number_op_qubits.append(system.fermionic_modes[index])
            if type == 2:
                number_op_controls.append(1)
            elif type == 3:
                number_op_controls.append(0)

    # Use left-elbow to store temporary logical AND of parity qubits and control
    block_encoding_metrics.add_to_clean_ancillae_usage(len(active_indices) - 1)
    block_encoding_metrics.number_of_elbows += len(active_indices) - 1

    temporary_qbool = clean_ancillae[clean_ancillae_index]
    # for i, active_mode in enumerate(non_number_op_indices):

    clean_ancillae_index += 1
    temporary_computations.append(
        cirq.Moment(
            cirq.X.on(temporary_qbool).controlled_by(
                *[system.fermionic_modes[i] for i in non_number_op_indices],
                *number_op_qubits,
                control_values=(
                    [int(not i) for i in non_number_op_types]
                )  # controlled by occupancies of the active modes
                + number_op_controls,
            )
        )
    )

    gates += temporary_computations

    # Flip block-encoding ancilla
    block_encoding_metrics.add_to_clean_ancillae_usage(1)
    block_encoding_metrics.add_to_clean_ancillae_usage(-1)
    block_encoding_metrics.number_of_elbows += 1
    gates.append(
        cirq.Moment(
            cirq.X.on(block_encoding_ancilla).controlled_by(
                *ctrls[0], temporary_qbool, control_values=ctrls[1] + [0]
            )
        )
    )
    block_encoding_metrics.add_to_clean_ancillae_usage(-(len(active_indices) - 1))

    # Reset clean ancillae
    gates += temporary_computations[::-1]

    # Update system
    for active_mode in non_number_op_indices:
        op_gates, op_metrics = _apply_fermionic_ladder_op(
            system, active_mode, ctrls=ctrls
        )
        gates += op_gates
        block_encoding_metrics += op_metrics

    return gates, block_encoding_metrics


def fermionic_plus_hc_block_encoding(
    system,
    block_encoding_ancillae,
    active_indices,
    operator_types,
    sign=1,
    clean_ancillae=[],
    ctrls=([], []),
):
    """Obtain operations for a block-encoding circuit for a product of fermionic ladder ops plus hermitian conjugate

    NOTE: Input arguements should correspond to only one term. The form of the hermitian conjugate will be inferred.

    TODO: Include operators of the form b_i b_i^

    Args:
        - system (lobe.system.System): The system object holding the system registers
        - block_encoding_ancillae (List[cirq.LineQubit]): The single block-encoding ancilla qubit
        - active_indices (List[int]): A list of the modes upon which the ladder operators act. Assumed to be in order
            of which operators are applied (right to left).
        - operator_types (List[int]): A list of ints indicating the type of ladder operators acting on each mode. Set to
            0 if operator is a annihilation/lowering ladder operator. 1 if creation/raising ladder operator. 2 if
            number operator
        - sign (int): Either 1 or -1 to indicate the sign of the term
        - clean_ancillae (List[cirq.LineQubit]): A list of qubits that are promised to start and end in the 0-state.
        - ctrls (Tuple(List[cirq.LineQubit], List[int])): A set of qubits and integers that correspond to
            the control qubits and values.

    Returns:
        - List of cirq operations representing the gates to be applied in the circuit
        - CircuitMetrics object representing cost of block-encoding circuit
    """
    assert len(ctrls[0]) == 1
    assert ctrls[1] == [1]
    assert len(block_encoding_ancillae) == 1
    block_encoding_ancilla = block_encoding_ancillae[0]
    block_encoding_metrics = CircuitMetrics()
    gates = []

    if sign == -1:
        gates.append(cirq.Z.on(ctrls[0][0]))

    if len(active_indices) == 1:
        assert (operator_types[0] == 0) or (operator_types[0] == 1)
        _gates, _metrics = _apply_fermionic_ladder_op(
            system, active_indices[0], ctrls=ctrls
        )
        gates += _gates
        block_encoding_metrics += _metrics
        return gates, block_encoding_metrics

    temporary_computations = []
    parity_qubits = []
    clean_ancillae_index = 0

    non_number_op_indices = []
    non_number_op_types = []
    number_op_qubits = []
    for type, index in zip(operator_types, active_indices):
        if type != 2:
            non_number_op_types.append(type)
            non_number_op_indices.append(index)
        else:
            number_op_qubits.append(system.fermionic_modes[index])

    for i, active_mode in enumerate(non_number_op_indices[:-1]):
        parity_qubit = system.fermionic_modes[active_mode]
        parity_qubits.append(parity_qubit)
        clean_ancillae_index += 1

        if non_number_op_types[i]:
            temporary_computations.append(cirq.Moment(cirq.X.on(parity_qubit)))

        temporary_computations.append(
            cirq.Moment(
                cirq.X.on(parity_qubit).controlled_by(
                    system.fermionic_modes[non_number_op_indices[i + 1]],
                    control_values=[not non_number_op_types[i + 1]],
                )
            )
        )

    # Use left-elbow to store temporary logical AND of parity qubits
    block_encoding_metrics.add_to_clean_ancillae_usage(
        len(parity_qubits) + len(number_op_qubits) - 1
    )
    block_encoding_metrics.number_of_elbows += (
        len(parity_qubits) + len(number_op_qubits) - 1
    )
    temporary_qbool = clean_ancillae[clean_ancillae_index]
    temporary_computations.append(
        cirq.Moment(
            cirq.X.on(temporary_qbool).controlled_by(
                *parity_qubits,
                *number_op_qubits,
                control_values=([0] * len(parity_qubits))
                + ([1] * len(number_op_qubits)),
            )
        )
    )
    gates += temporary_computations

    # Flip block-encoding ancilla
    block_encoding_metrics.add_to_clean_ancillae_usage(1)
    block_encoding_metrics.add_to_clean_ancillae_usage(-1)
    block_encoding_metrics.number_of_elbows += 1
    gates.append(
        cirq.Moment(
            cirq.X.on(block_encoding_ancilla).controlled_by(
                *ctrls[0], temporary_qbool, control_values=ctrls[1] + [0]
            )
        )
    )
    block_encoding_metrics.add_to_clean_ancillae_usage(
        -(len(parity_qubits) + len(number_op_qubits) - 1)
    )

    # Reset clean ancillae
    gates += temporary_computations[::-1]

    # Update system
    number_of_swaps = math.comb(len(non_number_op_indices), 2)
    if number_of_swaps % 2:
        sign_qubit = system.fermionic_modes[non_number_op_indices[0]]
        if not non_number_op_types[0]:
            gates.append(cirq.Moment(cirq.X.on(sign_qubit)))
        gates.append(
            cirq.Moment(
                cirq.Z.on(sign_qubit).controlled_by(
                    *ctrls[0],
                    control_values=ctrls[1],
                )
            )
        )
        if not non_number_op_types[0]:
            gates.append(cirq.Moment(cirq.X.on(sign_qubit)))

    for active_mode in non_number_op_indices:
        op_gates, op_metrics = _apply_fermionic_ladder_op(
            system, active_mode, ctrls=ctrls
        )
        gates += op_gates
        block_encoding_metrics += op_metrics

    return gates, block_encoding_metrics


def _apply_fermionic_ladder_op(system, index, ctrls=([], [])):
    """Apply the controlled $\\vec{Z}X$ operator to apply a fermionic ladder operator to the system.

    Args:
        - system (lobe.system.System): The system object holding the system registers
        - index (int): The mode index upon which the ladder operator acts
        - ctrls (Tuple(List[cirq.LineQubit], List[int])): A set of qubits and integers that correspond to
            the control qubits and values.

    Returns:
        - List of cirq operations representing the gates to be applied in the circuit
        - CircuitMetrics object representing cost of block-encoding circuit
    """
    operator_metrics = CircuitMetrics()
    gates = []

    for system_qubit in system.fermionic_modes[:index]:
        gates.append(
            cirq.Moment(
                cirq.Z.on(system_qubit).controlled_by(
                    *ctrls[0], control_values=ctrls[1]
                )
            )
        )
    gates.append(
        cirq.Moment(
            cirq.X.on(system.fermionic_modes[index]).controlled_by(
                *ctrls[0], control_values=ctrls[1]
            )
        )
    )

    return gates, operator_metrics
