import cirq


def add_select_oracle(circuit, validation, control, index_register, system, operators):
    for operator_index, operator in enumerate(operators):
        if operator[0] == operator[1]:
            circuit = _add_no_swap_unitary(
                circuit, validation, index_register, operator_index, system[operator[0]]
            )
        else:
            circuit = _add_swap_unitary(
                circuit,
                validation,
                control,
                index_register,
                operator_index,
                system,
                operator,
            )

    return circuit


def _add_no_swap_unitary(
    circuit, validation, index_register, operator_index, system_qubit
):
    index_register_control_values = [
        int(i) for i in format(operator_index, f"#0{2+len(index_register)}b")[2:]
    ][::-1]
    control_values = index_register_control_values + [1]
    control_qubits = index_register + [system_qubit]
    circuit.append(
        cirq.X.on(validation).controlled_by(
            *control_qubits, control_values=control_values
        )
    )
    return circuit


def _add_swap_unitary(
    circuit, validation, control, index_register, operator_index, system, operator
):
    index_register_control_values = [
        int(i) for i in format(operator_index, f"#0{2+len(index_register)}b")[2:]
    ][::-1]
    control_values = [1] + index_register_control_values + [0, 1]
    control_qubits = (
        [validation] + index_register + [system[operator[0]]] + [system[operator[1]]]
    )

    circuit.append(
        cirq.X.on(control).controlled_by(*control_qubits, control_values=control_values)
    )
    circuit.append(cirq.X.on(system[operator[0]]).controlled_by(control))
    circuit.append(cirq.X.on(system[operator[1]]).controlled_by(control))

    circuit.append(cirq.X.on(validation).controlled_by(control))

    circuit.append(
        cirq.X.on(control).controlled_by(
            *([validation] + index_register),
            control_values=[0] + index_register_control_values,
        )
    )

    return circuit
