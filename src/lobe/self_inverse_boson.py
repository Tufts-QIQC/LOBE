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
    gates = []
    

    unitary_index_qubit = block_encoding_ancillae[0] #qubit that indexes S vs. S^\dagger
    rotation_qubit = block_encoding_ancillae[1]
    if sign == -1:
        gates += _apply_negative_identity(unitary_index_qubit, ctrls=ctrls)

    gates.append(cirq.H.on(unitary_index_qubit))


    left_elbow, _ = decompose_controls_left(
        (ctrls[0] + [unitary_index_qubit], ctrls[1] + [1]), clean_ancillae[0]
    )
    gates.append(left_elbow)
    gates.append(
        cirq.X.on(rotation_qubit).controlled_by(clean_ancillae[0])
    )

    rotation_gates, _ = _add_multi_bosonic_rotations(
        rotation_qubit,
        system.bosonic_modes[active_mode],
        creation_exponent=1,
        annihilation_exponent=1,
        clean_ancillae=clean_ancillae,
        ctrls=([], []),
    )
    gates.append(rotation_gates)
    gates.append(
        cirq.X.on(rotation_qubit).controlled_by(clean_ancillae[0])
    )
    right_elbow, _ = decompose_controls_right(
            (ctrls[0] + [unitary_index_qubit], ctrls[1] + [1]), clean_ancillae[0]
    )
    gates.append(right_elbow)
  
    gates.append(cirq.X.on(unitary_index_qubit))
    gates.append(cirq.H.on(unitary_index_qubit))


    return gates

def self_inverse_bosonic_product_plus_hc_block_encoding(
    system,
    block_encoding_ancillae,
    sign=1,
    clean_ancillae=[],
    ctrls=([], []),
):
    """
    A self inverse block-encoding of the operator a_i a_j a_k... + h.c.
    Args:
        - system (lobe.system.System): The system object holding the system registers
        - block_encoding_ancillae (List[cirq.LineQubit]): The block-encoding ancillae qubits
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
    
    rotation_qubit = block_encoding_ancillae[0]
    unitary_index_qubit = block_encoding_ancillae[1] # qubit that indexes S vs. S^\dagger
    operator_index = block_encoding_ancillae[2] #qubit that indexes a_i vs. a_i^\dagger

    if sign == -1:
        gates += _apply_negative_identity(rotation_qubit, ctrls=ctrls)

    gates.append(cirq.H.on(unitary_index_qubit))  
    gates.append(cirq.H.on(operator_index))

    gates.append(cirq.X.on(operator_index).controlled_by(unitary_index_qubit))  

    adder_gates, adder_metrics = add_classical_value(
        list(system.bosonic_modes[0]),
        1,
        clean_ancillae=clean_ancillae,
        ctrls=([operator_index], [0]),
    )
    gates += adder_gates
    block_encoding_metrics += adder_metrics

    gates.append(cirq.X.on(rotation_qubit).controlled_by(unitary_index_qubit))  
    rotation_gates, rotation_metrics = _add_multi_bosonic_rotations(
        rotation_qubit,
        list(system.bosonic_modes[0]),
        1,
        0,
        clean_ancillae=clean_ancillae,
        ctrls=ctrls,
    )
    gates += rotation_gates
    block_encoding_metrics += rotation_metrics
    gates.append(cirq.X.on(rotation_qubit).controlled_by(unitary_index_qubit))  

    adder_gates, adder_metrics = add_classical_value(
        list(system.bosonic_modes[0]),
        -1,
        clean_ancillae=clean_ancillae,
        ctrls=([operator_index], [1]),
    )
    gates += adder_gates
    block_encoding_metrics += adder_metrics
    
    gates.append(cirq.X.on(operator_index).controlled_by(unitary_index_qubit))  

        
    gates.append(cirq.X.on(unitary_index_qubit))  
    gates.append(cirq.H.on(unitary_index_qubit))  
    gates.append(cirq.H.on(operator_index))

    return gates, block_encoding_metrics
