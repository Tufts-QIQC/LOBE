import cirq
import numpy as np
from .addition import add_classical_value
from .decompose import decompose_controls_left, decompose_controls_right
from .metrics import CircuitMetrics
from .multiplexed_rotations import get_decomposed_multiplexed_rotation_circuit
from ._utils import _apply_negative_identity
from .bosonic import _add_multi_bosonic_rotations


def self_inverse_bosonic_number_operator_block_encoding(
        system,
        block_encoding_ancillae,
        active_mode,
        sign=1,
        clean_ancillae=[],
        ctrls=([], []),
):
    """
    A self inverse block-encoding of the operator a_i^\dagger a_i
    Args:
        - system (lobe.system.System): The system object holding the system registers
        - block_encoding_ancillae (List[cirq.LineQubit]): The block-encoding ancillae qubits
        - active_mode (int): The mode upon which the ladder operator acts.
        - sign (int): Either 1 or -1 to indicate the sign of the term
        - clean_ancillae (List[cirq.LineQubit]): A list of qubits that are promised to start and end in the 0-state.
        - ctrls (Tuple(List[cirq.LineQubit], List[int])): A set of qubits and integers that correspond to
            the control qubits and values.

     Returns:
        - List of cirq operations representing the gates to be applied in the circuit
        - CircuitMetrics object representing cost of block-encoding circuit
    """
    #TODO: Add block encoding metrics
    gates = []
    

    unitary_index_qubit = block_encoding_ancillae[0] #qubit that indexes S vs. S^\dagger
    rotation_qubit = block_encoding_ancillae[1]
    if sign == -1:
        gates += _apply_negative_identity(unitary_index_qubit, ctrls=ctrls)

    gates.append(cirq.H.on(unitary_index_qubit))



    gates.append(
        cirq.X.on(rotation_qubit).controlled_by(unitary_index_qubit)
    )

    rotation_gates, _ = _add_multi_bosonic_rotations(
        rotation_qubit,
        system.bosonic_modes[active_mode],
        creation_exponent=1,
        annihilation_exponent=1,
        clean_ancillae=clean_ancillae,
        ctrls=ctrls,
    )
    gates.append(rotation_gates)
    gates.append(
        cirq.X.on(rotation_qubit).controlled_by(unitary_index_qubit)
    )

  
    gates.append(cirq.X.on(unitary_index_qubit))
    gates.append(cirq.H.on(unitary_index_qubit))


    return gates


def self_inverse_bosonic_product_plus_hc_block_encoding(
    system,
    block_encoding_ancillae,
    active_indices,
    exponents_list,
    sign=1,
    clean_ancillae=[],
    ctrls=([], []),
):
    """
    A self inverse block-encoding of the operator 
    $(a_i^\dagger)^{R_i} (a_i)^{S_i}) (a_j^\dagger)^{R_j} (a_j)^{S_j}) ... (a_l^\dagger)^{R_l} (a_l)^{S_l})$ + h.c.
    Args:
        - system (lobe.system.System): The system object holding the system registers
        - block_encoding_ancillae (List[cirq.LineQubit]): The block-encoding ancillae qubits
        - active_indices (List[int]): A list of the modes upon which the ladder operators act. Assumed to be in order
            of which operators are applied (right to left).
        - exponents_list (List[tuple]): A list of tuples (Ri, Si) containing the number of creation (Ri) and
            annihilation (Si) operators in the operator acting on mode i.
        - sign (int): Either 1 or -1 to indicate the sign of the term
        - clean_ancillae (List[cirq.LineQubit]): A list of qubits that are promised to start and end in the 0-state.
        - ctrls (Tuple(List[cirq.LineQubit], List[int])): A set of qubits and integers that correspond to
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
    
    unitary_index_qubit = block_encoding_ancillae[0] # qubit that indexes S vs. S^\dagger
    operator_index = block_encoding_ancillae[1] #qubit that indexes a_l vs. a_l^\dagger
    rotation_register = block_encoding_ancillae[2:] #rot_i, rot_j, .. (one per active mode)

    if sign == -1:
        gates += _apply_negative_identity(operator_index, ctrls=ctrls)

    gates.append(cirq.H.on(unitary_index_qubit))  
    gates.append(cirq.H.on(operator_index))

    gates.append(cirq.X.on(operator_index).controlled_by(unitary_index_qubit))
    _gates, _metrics = decompose_controls_left(
        (ctrls[0] + [operator_index], ctrls[1] + [0]), clean_ancillae[0]
    )
    gates += _gates
    block_encoding_metrics += _metrics

    for i, active_index in enumerate(active_indices):
        Ri, Si = exponents_list[i][0], exponents_list[i][1]

        gates.append(cirq.X.on(rotation_register[i]).controlled_by(unitary_index_qubit))

        adder_gates, adder_metrics = add_classical_value(
            system.bosonic_modes[active_index],
            Ri-Si,
            clean_ancillae=clean_ancillae[1:],
            ctrls = ([clean_ancillae[0]], [1]),
        )    
        gates.append(adder_gates)
        block_encoding_metrics += _metrics

        rotation_gates, rotation_metrics = _add_multi_bosonic_rotations(
            rotation_register[i],
            system.bosonic_modes[active_index],
            Ri,
            Si,
            clean_ancillae=clean_ancillae[1:],
            ctrls=ctrls,
        )
        gates += rotation_gates
        block_encoding_metrics += rotation_metrics

    gates.append(
        cirq.X.on(clean_ancillae[0]).controlled_by(*ctrls[0], control_values=ctrls[1])
    ) # right elbow followed by left elbow is a CNOT

    for i, active_index in enumerate(active_indices):
        Ri, Si = exponents_list[i][0], exponents_list[i][1]
        adder_gates, adder_metrics = add_classical_value(
            system.bosonic_modes[active_index],
            -Ri + Si,
            clean_ancillae=clean_ancillae[1:],
            ctrls=([clean_ancillae[0]], [1]),
        )
        gates += adder_gates
        block_encoding_metrics += adder_metrics

        gates.append(cirq.X.on(rotation_register[i]).controlled_by(unitary_index_qubit))

    _gates, _metrics = decompose_controls_right(
        (ctrls[0] + [operator_index], ctrls[1] + [1]), clean_ancillae[0]
    )
    gates += _gates
    block_encoding_metrics += _metrics

    
    

    gates.append(cirq.X.on(operator_index).controlled_by(unitary_index_qubit))
        
    gates.append(cirq.X.on(unitary_index_qubit))  
    gates.append(cirq.H.on(unitary_index_qubit))  
    gates.append(cirq.H.on(operator_index))

    return gates, block_encoding_metrics