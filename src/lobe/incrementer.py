import cirq
import numpy as np


def add_classical_value(
    register,
    classical_value,
    clean_ancillae,
    ctrls=([], []),
    compute_cost=False,
):
    if compute_cost:
        assert len(ctrls[0]) == 1
    classical_value = classical_value % (1 << len(register))

    if classical_value == 0:
        if compute_cost:
            return [], 0
        return []

    options = [(classical_value, False), ((1 << len(register)) - classical_value, True)]

    outcomes = []
    costs = []
    for value, flip_decr in options:
        decrement = False
        if value < 0:
            decrement = True
        if flip_decr:
            decrement = not decrement

        num_levels = int(np.ceil(np.log2(np.abs(value)))) + 1
        gates = []
        value = np.abs(value)
        num_toffs = 0
        for level in range(num_levels - 1, -1, -1):

            qubits_involved = register
            if level > 0:
                qubits_involved = register[:-level]

            if (1 << level) <= value:
                num_toffs += len(qubits_involved) - 1
                gates += add_incrementer(
                    [],
                    qubits_involved,
                    clean_ancillae[: len(qubits_involved) - 2],
                    decrement=decrement,
                    control_register=ctrls[0],
                    control_values=ctrls[1],
                )
                value -= 1 << level
        outcomes.append(gates)
        costs.append(num_toffs)

    if compute_cost:
        if costs[1] < costs[0]:
            return outcomes[1], costs[1]
        return outcomes[0], costs[0]
    if len(outcomes[1]) < len(outcomes[0]):
        return outcomes[1]
    return outcomes[0]


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
