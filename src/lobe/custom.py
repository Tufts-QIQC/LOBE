from .metrics import CircuitMetrics
from .addition import add_classical_value_gate_efficient
from .multiplexed_rotations import get_decomposed_multiplexed_rotation_circuit
from .bosonic import _get_bosonic_rotation_angles
from .fermionic import _apply_fermionic_ladder_op
import cirq
import numpy as np


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

    NOTE: Term is expected to be in the form: $b_i b_j a_k^\dagger a_l^\dagger + b_j^\dagger b_i^\dagger a_k a_l$.
        Expected ordering of active indices is [l, k, j, i].

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
    assert len(ctrls[0]) == 1
    assert ctrls[1] == [1]
    gates = []
    block_encoding_metrics = CircuitMetrics()

    block_encoding_metrics.add_to_clean_ancillae_usage(1)
    temporary_qbool = clean_ancillae[0]
    gates.append(
        cirq.X.on(temporary_qbool).controlled_by(
            system.fermionic_register[active_indices[2]]
        )
    )

    for be_ancilla, bosonic_index in zip(
        block_encoding_ancillae[:2], active_indices[:2]
    ):
        block_encoding_metrics.add_to_clean_ancillae_usage(
            2 * len(system.bosonic_system[bosonic_index])
        )
        block_encoding_metrics.number_of_elbows += len(
            system.bosonic_system[bosonic_index]
        )
        gates += add_classical_value_gate_efficient(
            system.bosonic_system[bosonic_index],
            1,
            clean_ancillae=clean_ancillae[1:],
            ctrls=(ctrls[0] + [temporary_qbool], ctrls[1] + [1]),
        )
        block_encoding_metrics.add_to_clean_ancillae_usage(
            -(2 * len(system.bosonic_system[bosonic_index]))
        )

        rotation_angles = _get_bosonic_rotation_angles(
            system.maximum_occupation_number, 0, 1
        )
        block_encoding_metrics.number_of_rotations += len(rotation_angles) + 1
        gates += get_decomposed_multiplexed_rotation_circuit(
            system.bosonic_system[bosonic_index] + [be_ancilla],
            rotation_angles,
            clean_ancillae=clean_ancillae[1:],
            ctrls=ctrls,
        )

        block_encoding_metrics.add_to_clean_ancillae_usage(
            2 * len(system.bosonic_system[bosonic_index])
        )
        block_encoding_metrics.number_of_elbows += len(
            system.bosonic_system[bosonic_index]
        )
        gates += add_classical_value_gate_efficient(
            system.bosonic_system[bosonic_index],
            -1,
            clean_ancillae=clean_ancillae[1:],
            ctrls=(ctrls[0] + [temporary_qbool], ctrls[1] + [0]),
        )
        block_encoding_metrics.add_to_clean_ancillae_usage(
            -(2 * len(system.bosonic_system[bosonic_index]))
        )

    gates.append(
        cirq.X.on(temporary_qbool).controlled_by(
            system.fermionic_register[active_indices[3]]
        )
    )
    block_encoding_metrics.number_of_elbows += 1
    block_encoding_metrics.add_to_clean_ancillae_usage(1)
    gates.append(
        cirq.X.on(block_encoding_ancillae[2]).controlled_by(
            *ctrls[0], temporary_qbool, control_values=ctrls[1] + [1]
        )
    )
    block_encoding_metrics.add_to_clean_ancillae_usage(-1)
    gates.append(
        cirq.X.on(temporary_qbool).controlled_by(
            system.fermionic_register[active_indices[3]]
        )
    )
    gates.append(
        cirq.X.on(temporary_qbool).controlled_by(
            system.fermionic_register[active_indices[2]]
        )
    )

    gates.append(
        cirq.Z.on(ctrls[0][0]).controlled_by(
            system.fermionic_register[active_indices[2]], control_values=[0]
        )
    )

    op_gates, op_metrics = _apply_fermionic_ladder_op(
        system, active_indices[2], ctrls=ctrls
    )
    gates += op_gates
    block_encoding_metrics += op_metrics
    op_gates, op_metrics = _apply_fermionic_ladder_op(
        system, active_indices[3], ctrls=ctrls
    )
    gates += op_gates
    block_encoding_metrics += op_metrics

    return gates, block_encoding_metrics


def yukawa_3point_pair_term_block_encoding(
    system,
    block_encoding_ancillae,
    active_indices,
    clean_ancillae=[],
    ctrls=([], []),
):
    """Add a block-encoding circuit for the 3 point pair term in the full Yukawa model

    NOTE: Term is expected to be in the form: $b_i b_j a_k^\dagger + b_j^\dagger b_i^\dagger a_k$.
        Expected ordering of active indices is [k, j, i].

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
    assert len(ctrls[0]) == 1
    assert ctrls[1] == [1]
    gates = []
    block_encoding_metrics = CircuitMetrics()

    block_encoding_metrics.add_to_clean_ancillae_usage(1)
    temporary_qbool = clean_ancillae[0]
    gates.append(
        cirq.X.on(temporary_qbool).controlled_by(
            system.fermionic_register[active_indices[1]]
        )
    )

    block_encoding_metrics.add_to_clean_ancillae_usage(
        2 * len(system.bosonic_system[active_indices[0]])
    )
    block_encoding_metrics.number_of_elbows += len(
        system.bosonic_system[active_indices[0]]
    )
    gates += add_classical_value_gate_efficient(
        system.bosonic_system[active_indices[0]],
        1,
        clean_ancillae=clean_ancillae[1:],
        ctrls=(ctrls[0] + [temporary_qbool], ctrls[1] + [1]),
    )
    block_encoding_metrics.add_to_clean_ancillae_usage(
        -(2 * len(system.bosonic_system[active_indices[0]]))
    )

    rotation_angles = _get_bosonic_rotation_angles(
        system.maximum_occupation_number, 0, 1
    )
    block_encoding_metrics.number_of_rotations += len(rotation_angles) + 1
    gates += get_decomposed_multiplexed_rotation_circuit(
        system.bosonic_system[active_indices[0]] + [block_encoding_ancillae[0]],
        rotation_angles,
        clean_ancillae=clean_ancillae[1:],
        ctrls=ctrls,
    )

    block_encoding_metrics.add_to_clean_ancillae_usage(
        2 * len(system.bosonic_system[active_indices[0]])
    )
    block_encoding_metrics.number_of_elbows += len(
        system.bosonic_system[active_indices[0]]
    )
    gates += add_classical_value_gate_efficient(
        system.bosonic_system[active_indices[0]],
        -1,
        clean_ancillae=clean_ancillae[1:],
        ctrls=(ctrls[0] + [temporary_qbool], ctrls[1] + [0]),
    )
    block_encoding_metrics.add_to_clean_ancillae_usage(
        -(2 * len(system.bosonic_system[active_indices[0]]))
    )

    gates.append(
        cirq.X.on(temporary_qbool).controlled_by(
            system.fermionic_register[active_indices[2]]
        )
    )
    block_encoding_metrics.number_of_elbows += 1
    block_encoding_metrics.add_to_clean_ancillae_usage(1)
    gates.append(
        cirq.X.on(block_encoding_ancillae[1]).controlled_by(
            *ctrls[0], temporary_qbool, control_values=ctrls[1] + [1]
        )
    )
    block_encoding_metrics.add_to_clean_ancillae_usage(-1)
    gates.append(
        cirq.X.on(temporary_qbool).controlled_by(
            system.fermionic_register[active_indices[2]]
        )
    )
    gates.append(
        cirq.X.on(temporary_qbool).controlled_by(
            system.fermionic_register[active_indices[1]]
        )
    )

    gates.append(
        cirq.Z.on(ctrls[0][0]).controlled_by(
            system.fermionic_register[active_indices[1]], control_values=[0]
        )
    )

    op_gates, op_metrics = _apply_fermionic_ladder_op(
        system, active_indices[1], ctrls=ctrls
    )
    gates += op_gates
    block_encoding_metrics += op_metrics
    op_gates, op_metrics = _apply_fermionic_ladder_op(
        system, active_indices[2], ctrls=ctrls
    )
    gates += op_gates
    block_encoding_metrics += op_metrics

    return gates, block_encoding_metrics


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
