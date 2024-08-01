import cirq


def add_incrementer(
    circuit,
    register,
    clean_ancilla,
    decrement=False,
    control_register=[],
    control_values=[],
):
    """This function adds an oracle to our circuit that increments the binary value of the register by +/- 1.

    Implementation: https://algassert.com/circuits/2015/06/12/Constructing-Large-Increment-Gates.html
    """
    circuit = []
    if len(control_values) != len(control_register):
        control_values += [1] * (len(control_register) - len(control_values))

    if len(register) == 1:
        circuit.append(
            cirq.Moment(
                cirq.X.on(register[0]).controlled_by(
                    *control_register, control_values=control_values
                )
            )
        )
        return circuit

    if len(register) == 2:
        if decrement:
            circuit.append(cirq.Moment(cirq.X.on_each(*register)))

        circuit.append(
            cirq.Moment(
                cirq.X.on(register[0]).controlled_by(
                    register[1], *control_register, control_values=[1] + control_values
                )
            )
        )
        circuit.append(
            cirq.Moment(
                cirq.X.on(register[1]).controlled_by(
                    *control_register, control_values=control_values
                )
            )
        )

        if decrement:
            circuit.append(cirq.Moment(cirq.X.on_each(*register)))

        return circuit

    assert len(clean_ancilla) == len(register) - 2
    flipped_register = register[::-1]  # follow same ordering as in blog post

    if decrement:
        circuit.append(cirq.Moment(cirq.X.on_each(*register)))

    # n-2 Toffolis going down
    first_control = flipped_register[0]
    second_control = flipped_register[1]
    counter = 0
    while counter < len(register) - 2:
        circuit.append(
            cirq.Moment(
                cirq.X.on(clean_ancilla[counter]).controlled_by(
                    first_control, second_control
                )
            )
        )
        first_control = clean_ancilla[counter]
        second_control = flipped_register[counter + 2]
        counter += 1

    circuit.append(
        cirq.Moment(
            cirq.X.on(flipped_register[-1]).controlled_by(
                clean_ancilla[-1],
                *control_register,
                control_values=[1] + control_values
            )
        )
    )

    counter -= 1
    first_control = clean_ancilla[counter - 1]
    second_control = flipped_register[counter + 1]

    while counter > 0:
        circuit.append(
            cirq.Moment(
                cirq.X.on(clean_ancilla[counter]).controlled_by(
                    first_control, second_control
                )
            )
        )
        circuit.append(
            cirq.Moment(
                cirq.X.on(second_control).controlled_by(
                    first_control,
                    *control_register,
                    control_values=[1] + control_values
                )
            )
        )
        counter -= 1
        first_control = clean_ancilla[counter - 1]
        second_control = flipped_register[counter + 1]

    circuit.append(
        cirq.Moment(
            cirq.X.on(clean_ancilla[0]).controlled_by(
                flipped_register[0], flipped_register[1]
            )
        )
    )
    circuit.append(
        cirq.Moment(
            cirq.X.on(flipped_register[1]).controlled_by(
                flipped_register[0],
                *control_register,
                control_values=[1] + control_values
            )
        )
    )
    circuit.append(
        cirq.Moment(
            cirq.X.on(flipped_register[0]).controlled_by(
                *control_register, control_values=control_values
            )
        )
    )

    if decrement:
        circuit.append(cirq.Moment(cirq.X.on_each(*register)))

    return circuit
