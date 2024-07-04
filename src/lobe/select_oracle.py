import cirq
import numpy as np


def add_select_oracle(circuit, validation, control, index_register, system, operators):
    """Add the select oracle for LOBE into the quantum circuit.

    Args:
        circuit (cirq.Circuit): The quantum circuit onto which the select oracle will be added
        validation (cirq.LineQubit): The validation qubit
        control (cirq.LineQubit): The ancilla qubit that is used for elbows
        index_register (List[cirq.LineQubit]): The qubit register that is used to index the operators
        system (List[cirq.LineQubit]): The qubit register that is used to encode the system
        operators (List[List[LadderOperator]]): The ladder operators included in the Hamiltonian.
            Each item in the list is a list of LadderOperators and corresponds to a term comprising several
            ladder operators.
            The first integer in this tuple dictates the type of operator . The second integer in this tuple corresponds to the mode that this ladder operator acts on.

    Returns:
        cirq.Circuit: The updated quantum circuit
    """
    for operator_index, operator in enumerate(operators):
        op_types = [ladder_op.particle_type for ladder_op in operator]
        if np.allclose(op_types, 0):  # All terms are fermionic terms
            if len(operator) == 2 and operator[0].mode == operator[1].mode:
                circuit = _add_fermionic_particle_number_op(
                    circuit,
                    validation,
                    index_register,
                    operator_index,
                    system[-operator[0].mode - 1],
                )
            else:
                circuit = _add_fermionic_ladder_operator(
                    circuit,
                    validation,
                    control,
                    index_register,
                    operator_index,
                    system,
                    operator,
                )
    return circuit


def _add_fermionic_particle_number_op(
    circuit, validation, index_register, operator_index, system_qubit
):
    # Get binary control values corresponding to current operator index
    index_register_control_values = [
        int(i) for i in format(operator_index, f"#0{2+len(index_register)}b")[2:]
    ]
    control_values = index_register_control_values + [1]
    control_qubits = index_register + [system_qubit]
    # Flip validation to 0 to mark that we hit a state that statisfied this operator
    circuit.append(
        cirq.X.on(validation).controlled_by(
            *control_qubits, control_values=control_values
        )
    )
    return circuit


def _add_fermionic_ladder_operator(
    circuit, validation, control, index_register, operator_index, system, operator
):
    # Get binary control values corresponding to current operator index
    index_register_control_values = [
        int(i) for i in format(operator_index, f"#0{2+len(index_register)}b")[2:]
    ]
    control_values = (
        [1]
        + index_register_control_values
        + (
            [0] * (len(operator) // 2)
        )  # Make sure system qubits for creation ops are 0-ctrl's
        + (
            [1] * (len(operator) // 2)
        )  # Make sure system qubits for annihilation ops are 1-ctrl's
    )
    control_qubits = [validation] + index_register
    # Add system qubits to control register
    for ladder_op in operator:
        control_qubits += [system[-ladder_op.mode - 1]]

    # This is essentially a left-elbow
    circuit.append(
        cirq.X.on(control).controlled_by(*control_qubits, control_values=control_values)
    )

    # Reverse loop because operators act starting from the right
    for ladder_op in operator[::-1]:
        for system_qubit in system[::-1][: ladder_op.mode]:
            circuit.append(cirq.Z.on(system_qubit).controlled_by(control))
        circuit.append(cirq.X.on(system[-ladder_op.mode - 1]).controlled_by(control))

    circuit.append(cirq.X.on(validation).controlled_by(control))

    # This is essentially a right-elbow
    circuit.append(
        cirq.X.on(control).controlled_by(
            *([validation] + index_register),
            control_values=[0] + index_register_control_values,
        )
    )

    return circuit
