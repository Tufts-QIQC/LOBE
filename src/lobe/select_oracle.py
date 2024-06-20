import cirq


def add_select_oracle(circuit, validation, control, index_register, system, operators):
    for operator_index, operator in enumerate(operators):
        if len(operator) == 2:
            if operator[0] == operator[1]:
                circuit = _add_no_swap_unitary(
                    circuit,
                    validation,
                    index_register,
                    operator_index,
                    system[-operator[0] - 1],
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
        elif len(operator) == 4:
            circuit = _add_two_body_term(
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
    ]
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
    ]
    control_values = [1] + index_register_control_values + [0, 1]
    control_qubits = (
        [validation]
        + index_register
        + [system[-operator[0] - 1]]
        + [system[-operator[1] - 1]]
    )

    circuit.append(
        cirq.X.on(control).controlled_by(*control_qubits, control_values=control_values)
    )

    start = min(-operator[0] - 1, -operator[1] - 1)
    stop = max(-operator[0] - 1, -operator[1] - 1)

    circuit.append(cirq.X.on(system[stop]).controlled_by(control))

    for i in reversed(range(start + 1, stop)):
        circuit.append(cirq.Z.on(system[i]).controlled_by(control))

    circuit.append(cirq.X.on(system[start]).controlled_by(control))

    circuit.append(cirq.X.on(validation).controlled_by(control))

    circuit.append(
        cirq.X.on(control).controlled_by(
            *([validation] + index_register),
            control_values=[0] + index_register_control_values,
        )
    )

    return circuit


def _add_two_body_term(
    circuit, validation, control, index_register, operator_index, system, operator
):
    index_register_control_values = [
        int(i) for i in format(operator_index, f"#0{2+len(index_register)}b")[2:]
    ]
    control_values = [1] + index_register_control_values + [0, 0, 1, 1]
    control_qubits = (
        [validation]
        + index_register
        + [system[-operator[0] - 1]]
        + [system[-operator[1] - 1]]
        + [system[-operator[2] - 1]]
        + [system[-operator[3] - 1]]
    )

    circuit.append(
        cirq.X.on(control).controlled_by(*control_qubits, control_values=control_values)
    )

    for i in operator[::-1]:
        for system_qubit in system[i + 1 :]:
            circuit.append(cirq.Z.on(system_qubit).controlled_by(control))
        circuit.append(cirq.X.on(system[i]).controlled_by(control))

    circuit.append(cirq.X.on(validation).controlled_by(control))

    circuit.append(
        cirq.X.on(control).controlled_by(
            *([validation] + index_register),
            control_values=[0] + index_register_control_values,
        )
    )

    return circuit
