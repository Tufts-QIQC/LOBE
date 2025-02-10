from .metrics import CircuitMetrics
from .addition import add_classical_value
from .multiplexed_rotations import get_decomposed_multiplexed_rotation_circuit
from .bosonic import _get_bosonic_rotation_angles
from .fermionic import _apply_fermionic_ladder_op
from .decompose import decompose_controls_left, decompose_controls_right

import cirq


def _custom_fermionic_plus_nonhc_block_encoding(
    system,
    block_encoding_ancillae,
    active_indices,
    sign=1,
    clean_ancillae=[],
    ctrls=([], []),
):
    """Add a block-encoding circuit for the operator: $b_i b_j b_k^\dagger + b_j^\dagger b_i^\dagger b_k^\dagger$.

    NOTE: Input arguements should correspond to only one term. The form of the hermitian conjugate will be inferred.

    Args:
        system (lobe.system.System): The system object holding the system registers
        block_encoding_ancillae (List[cirq.LineQubit]): A list with a single block-encoding ancilla qubit
        active_indices (List[int]): A list of the modes upon which the ladder operators act. Assumed to be in order
            of which operators are applied (right to left): [k, j, i].
        sign (int): Either 1 or -1 to indicate the sign of the term
        clean_ancillae (List[cirq.LineQubit]): A list of qubits that are promised to start and end in the 0-state.
        ctrls (Tuple(List[cirq.LineQubit], List[int])): A set of qubits and integers that correspond to
            the control qubits and values.

    Returns:
        - List of cirq operations representing the gates to be applied in the circuit
        - CircuitMetrics object representing cost of block-encoding circuit
    """
    assert len(ctrls[0]) == 1
    assert ctrls[1] == [1]
    block_encoding_ancilla = block_encoding_ancillae[0]
    gates = []
    block_encoding_metrics = CircuitMetrics()

    if sign == -1:
        gates.append(cirq.Z.on(ctrls[0][0]))

    temporary_computations = []
    temporary_computations.append(
        cirq.X.on(system.fermionic_modes[active_indices[1]]).controlled_by(
            system.fermionic_modes[active_indices[2]]
        )
    )

    block_encoding_metrics.add_to_clean_ancillae_usage(1)
    block_encoding_metrics.number_of_elbows += 1
    temporary_computations.append(
        cirq.X.on(clean_ancillae[0]).controlled_by(
            system.fermionic_modes[active_indices[0]],
            system.fermionic_modes[active_indices[1]],
            control_values=[0, 0],
        )
    )

    gates += temporary_computations
    block_encoding_metrics.add_to_clean_ancillae_usage(1)
    block_encoding_metrics.number_of_elbows += 1
    gates.append(
        cirq.X.on(block_encoding_ancilla).controlled_by(
            clean_ancillae[0], *ctrls[0], control_values=[0] + ctrls[1]
        )
    )
    block_encoding_metrics.add_to_clean_ancillae_usage(-1)
    gates += temporary_computations[::-1]

    block_encoding_metrics.add_to_clean_ancillae_usage(-1)

    gates.append(
        cirq.Z.on(ctrls[0][0]).controlled_by(
            system.fermionic_modes[active_indices[2]], control_values=[1]
        )
    )
    op_gates, op_metrics = _apply_fermionic_ladder_op(
        system, active_indices[2], ctrls=ctrls
    )
    gates += op_gates
    block_encoding_metrics += op_metrics
    op_gates, op_metrics = _apply_fermionic_ladder_op(
        system, active_indices[1], ctrls=ctrls
    )
    gates += op_gates
    block_encoding_metrics += op_metrics
    op_gates, op_metrics = _apply_fermionic_ladder_op(
        system, active_indices[0], ctrls=ctrls
    )
    gates += op_gates
    block_encoding_metrics += op_metrics
    return gates, block_encoding_metrics


def _custom_term_block_encoding(
    system,
    block_encoding_ancillae,
    active_indices,
    sign=1,
    clean_ancillae=[],
    ctrls=([], []),
):
    """Add a block-encoding circuit for the operator: $b_i a_j + b_i^\dagger a_j^\dagger$.

    NOTE: Input arguements should correspond to only one term. The form of the hermitian conjugate will be inferred.

    Args:
        system (lobe.system.System): The system object holding the system registers
        block_encoding_ancillae (List[cirq.LineQubit]): A list with a single block-encoding ancilla qubit
        active_indices (List[int]): A list of the modes upon which the ladder operators act. Assumed to be in order
            of which operators are applied (right to left): [j, i].
        sign (int): Either 1 or -1 to indicate the sign of the term
        clean_ancillae (List[cirq.LineQubit]): A list of qubits that are promised to start and end in the 0-state.
        ctrls (Tuple(List[cirq.LineQubit], List[int])): A set of qubits and integers that correspond to
            the control qubits and values.

    Returns:
        - List of cirq operations representing the gates to be applied in the circuit
        - CircuitMetrics object representing cost of block-encoding circuit
    """
    assert len(ctrls[0]) == 1
    assert ctrls[1] == [1]
    gates = []
    block_encoding_ancilla = block_encoding_ancillae[0]
    block_encoding_metrics = CircuitMetrics()

    if sign == -1:
        gates.append(cirq.Z.on(ctrls[0][0]))

    _gates, _metrics = decompose_controls_left(
        (ctrls[0] + [system.fermionic_modes[active_indices[1]]], ctrls[1] + [0]),
        clean_ancillae[0],
    )
    gates += _gates
    block_encoding_metrics += _metrics

    _gates, _metrics = add_classical_value(
        system.bosonic_modes[active_indices[0]],
        1,
        clean_ancillae=clean_ancillae[1:],
        ctrls=([clean_ancillae[0]], [1]),
    )
    gates += _gates
    block_encoding_metrics += _metrics

    rotation_angles = _get_bosonic_rotation_angles(
        system.maximum_occupation_number, 1, 0
    )
    _gates, _metrics = get_decomposed_multiplexed_rotation_circuit(
        system.bosonic_modes[active_indices[0]],
        block_encoding_ancilla,
        rotation_angles,
        clean_ancillae=clean_ancillae[1:],
        ctrls=ctrls,
    )
    gates += _gates
    block_encoding_metrics += _metrics

    gates.append(
        cirq.X.on(clean_ancillae[0]).controlled_by(*ctrls[0], control_values=ctrls[1])
    )
    _gates, _metrics = add_classical_value(
        system.bosonic_modes[active_indices[0]],
        -1,
        clean_ancillae=clean_ancillae[1:],
        ctrls=([clean_ancillae[0]], [1]),
    )
    gates += _gates
    block_encoding_metrics += _metrics

    _gates, _metrics = decompose_controls_right(
        (ctrls[0] + [system.fermionic_modes[active_indices[1]]], ctrls[1] + [1]),
        clean_ancillae[0],
    )
    gates += _gates
    block_encoding_metrics += _metrics

    _gates, _metrics = _apply_fermionic_ladder_op(
        system, active_indices[1], ctrls=ctrls
    )
    gates += _gates
    block_encoding_metrics += _metrics
    return gates, block_encoding_metrics
