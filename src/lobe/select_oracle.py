import cirq


def add_select_oracle(circuit, validation, control, index_register, system, operators):
    for operator_index, operator in enumerate(operators):
        if len(operator) == 2:
            if operator[0] == operator[1]:
                circuit = _add_particle_number_op(
                    circuit,
                    validation,
                    index_register,
                    operator_index,
                    system[-operator[0] - 1],
                )
            else:
                circuit = _add_ladder_operator(
                    circuit,
                    validation,
                    control,
                    index_register,
                    operator_index,
                    system,
                    operator,
                )
        elif len(operator) == 4:
            circuit = _add_ladder_operator(
                circuit,
                validation,
                control,
                index_register,
                operator_index,
                system,
                operator,
            )
    return circuit


def _add_particle_number_op(
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


def _add_ladder_operator(
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
    for i in operator:
        control_qubits += [system[-i - 1]]

    # This is essentially a left-elbow
    circuit.append(
        cirq.X.on(control).controlled_by(*control_qubits, control_values=control_values)
    )

    # Reverse loop because operators act starting from the right
    for i in operator[::-1]:
        for system_qubit in system[::-1][:i]:
            circuit.append(cirq.Z.on(system_qubit).controlled_by(control))
        circuit.append(cirq.X.on(system[-i - 1]).controlled_by(control))

    circuit.append(cirq.X.on(validation).controlled_by(control))

    # This is essentially a right-elbow
    circuit.append(
        cirq.X.on(control).controlled_by(
            *([validation] + index_register),
            control_values=[0] + index_register_control_values,
        )
    )

    return circuit
