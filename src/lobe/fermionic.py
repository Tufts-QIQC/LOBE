from .metrics import CircuitMetrics
import cirq
import math


def fermionic_product_block_encoding(
    system,
    block_encoding_ancilla,
    active_indices,
    operator_types,
    clean_ancillae=[],
    ctrls=([], []),
):
    """Obtain operations for a block-encoding circuit for a product of fermionic ladder operators

    Args:
        system (lobe.system.System): The system object holding the system registers
        block_encoding_ancilla (cirq.LineQubit): The single block-encoding ancilla qubit
        active_indices (List[int]): A list of the modes upon which the ladder operators act. Assumed to be in order
            of which operators are applied (right to left).
        operator_types (List[int]): A list of ints indicating the type of ladder operators acting on each mode. Set to
            0 if operator is a annihilation/lowering ladder operator. 1 if creation/raising ladder operator. 2 if
            number operator
        clean_ancillae (List[cirq.LineQubit]): A list of qubits that are promised to start and end in the 0-state.
        ctrls (Tuple(List[cirq.LineQubit], List[int])): A set of qubits and integers that correspond to
            the control qubits and values.

    Returns:
        - List of cirq operations representing the gates to be applied in the circuit
        - CircuitMetrics object representing cost of block-encoding circuit
    """
    assert len(ctrls[0]) == 1
    assert ctrls[1] == [1]
    block_encoding_metrics = CircuitMetrics()

    gates = []

    temporary_computations = []
    clean_ancillae_index = 0

    non_number_op_indices = []
    non_number_op_types = []
    number_op_qubits = []
    for type, index in zip(operator_types, active_indices):
        if type != 2:
            non_number_op_types.append(type)
            non_number_op_indices.append(index)
        else:
            number_op_qubits.append(system.fermionic_register[index])

    # Use left-elbow to store temporary logical AND of parity qubits and control
    block_encoding_metrics.add_to_clean_ancillae_usage(
        len(non_number_op_types) - 1 + len(number_op_qubits) - 1
    )
    block_encoding_metrics.number_of_elbows += (
        len(non_number_op_types) - 1 + len(number_op_qubits) - 1
    )

    temporary_qbool = clean_ancillae[clean_ancillae_index]
    # for i, active_mode in enumerate(non_number_op_indices):

    clean_ancillae_index += 1
    temporary_computations.append(
        cirq.Moment(
            cirq.X.on(temporary_qbool).controlled_by(
                *[system.fermionic_register[i] for i in non_number_op_indices],
                *number_op_qubits,
                control_values=(
                    [int(not i) for i in non_number_op_types]
                )  # controlled by occupancies of the active modes
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
        -(len(non_number_op_types) - 1 + len(number_op_qubits) - 1)
    )

    # Reset clean ancillae
    gates += temporary_computations[::-1]

    # Update system
    for active_mode in non_number_op_indices:
        for system_qubit in system.fermionic_register[:active_mode]:
            gates.append(
                cirq.Moment(
                    cirq.Z.on(system_qubit).controlled_by(
                        *ctrls[0], control_values=ctrls[1]
                    )
                )
            )
        gates.append(
            cirq.Moment(
                cirq.X.on(system.fermionic_register[active_mode]).controlled_by(
                    *ctrls[0], control_values=ctrls[1]
                )
            )
        )

    return gates, block_encoding_metrics


def fermionic_plus_hc_block_encoding(
    system,
    block_encoding_ancilla,
    active_indices,
    operator_types,
    clean_ancillae=[],
    ctrls=([], []),
):
    """Obtain operations for a block-encoding circuit for a product of fermionic ladder ops plus hermitian conjugate

    NOTE: Input arguements should correspond to only one term. The form of the hermitian conjugate will be inferred.

    TODO: Include number operators

    Args:
        system (lobe.system.System): The system object holding the system registers
        block_encoding_ancilla (cirq.LineQubit): The single block-encoding ancilla qubit
        active_indices (List[int]): A list of the modes upon which the ladder operators act. Assumed to be in order
            of which operators are applied (right to left).
        operator_types (List[int]): A list of ints indicating the type of ladder operators acting on each mode. Set to
            0 if operator is a annihilation/lowering ladder operator. 1 if creation/raising ladder operator. 2 if
            number operator
        clean_ancillae (List[cirq.LineQubit]): A list of qubits that are promised to start and end in the 0-state.
        ctrls (Tuple(List[cirq.LineQubit], List[int])): A set of qubits and integers that correspond to
            the control qubits and values.

    Returns:
        - List of cirq operations representing the gates to be applied in the circuit
        - CircuitMetrics object representing cost of block-encoding circuit
    """
    assert len(ctrls[0]) == 1
    assert ctrls[1] == [1]
    block_encoding_metrics = CircuitMetrics()

    gates = []

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
            number_op_qubits.append(system.fermionic_register[index])

    for i, active_mode in enumerate(non_number_op_indices[:-1]):
        parity_qubit = clean_ancillae[clean_ancillae_index]
        parity_qubits.append(parity_qubit)
        clean_ancillae_index += 1

        temporary_computations.append(
            cirq.Moment(
                cirq.X.on(parity_qubit).controlled_by(
                    system.fermionic_register[active_mode],
                    control_values=[not non_number_op_types[i]],
                )
            )
        )
        temporary_computations.append(
            cirq.Moment(
                cirq.X.on(parity_qubit).controlled_by(
                    system.fermionic_register[non_number_op_indices[i + 1]],
                    control_values=[not non_number_op_types[i + 1]],
                )
            )
        )
    block_encoding_metrics.add_to_clean_ancillae_usage(len(parity_qubits))

    # Use left-elbow to store temporary logical AND of parity qubits and control
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
    block_encoding_metrics.add_to_clean_ancillae_usage(-len(parity_qubits))
    gates += temporary_computations[::-1]

    # Update system
    active_qubits = [
        system.fermionic_register[active_mode] for active_mode in non_number_op_indices
    ]
    number_of_swaps = math.comb(len(non_number_op_indices), 2)
    if number_of_swaps % 2:
        sign_qubit = system.fermionic_register[non_number_op_indices[0]]
        if non_number_op_types[0]:
            gates.append(cirq.Moment(cirq.X.on(sign_qubit)))
        gates.append(
            cirq.Moment(
                cirq.Z.on(sign_qubit).controlled_by(
                    *ctrls[0],
                    control_values=ctrls[1],
                )
            )
        )
        if non_number_op_types[0]:
            gates.append(cirq.Moment(cirq.X.on(sign_qubit)))

    for active_mode in non_number_op_indices[::-1]:
        for system_qubit in system.fermionic_register[:active_mode]:
            gates.append(
                cirq.Moment(
                    cirq.Z.on(system_qubit).controlled_by(
                        *ctrls[0], control_values=ctrls[1]
                    )
                )
            )
        gates.append(
            cirq.Moment(
                cirq.X.on(system.fermionic_register[active_mode]).controlled_by(
                    *ctrls[0], control_values=ctrls[1]
                )
            )
        )

    return gates, block_encoding_metrics
