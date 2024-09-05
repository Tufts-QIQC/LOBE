import cirq
import numpy as np


def add_classical_value_incrementers(
    register, classical_value, clean_ancillae, ctrls=([], [])
):
    classical_value = classical_value % (1 << len(register))

    if classical_value == 0:
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
                control_values=[1] + control_values,
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
                    control_values=[1] + control_values,
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
                control_values=[1] + control_values,
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


def _load_m(m_val, m_reg, ctrls=([], [])):
    gates = []
    bit_string = format(m_val, f"0{2+len(m_reg)}b")[2:]
    for i, bit in enumerate(bit_string):
        if i < len(m_reg):
            if bit == "1":
                gates.append(
                    cirq.Moment(
                        cirq.X.on(m_reg[i]).controlled_by(
                            *ctrls[0], control_values=ctrls[1]
                        )
                    )
                )
    return gates


def _qadd_helper_left(n_bit, m_bit, carry_in, carry_out):
    gates = []
    gates.append(cirq.Moment(cirq.X.on(m_bit).controlled_by(carry_in)))
    gates.append(cirq.Moment(cirq.X.on(n_bit).controlled_by(carry_in)))
    gates.append(cirq.Moment(cirq.X.on(carry_out).controlled_by(n_bit, m_bit)))
    gates.append(cirq.Moment(cirq.X.on(carry_out).controlled_by(carry_in)))
    return gates


def _qadd_helper_right(n_bit, m_bit, carry_in, carry_out):
    gates = []
    gates.append(cirq.Moment(cirq.X.on(carry_out).controlled_by(carry_in)))
    gates.append(cirq.Moment(cirq.X.on(carry_out).controlled_by(n_bit, m_bit)))
    gates.append(cirq.Moment(cirq.X.on(m_bit).controlled_by(carry_in)))
    gates.append(cirq.Moment(cirq.X.on(n_bit).controlled_by(m_bit)))
    return gates


def _quantum_addtion(n_register, m_register, clean_ancillae, recursion_level=0):
    gates = []
    if len(n_register) == 0:
        return gates
    elif len(n_register) == 1:
        return [
            cirq.Moment(cirq.X.on(n_register[0]).controlled_by(clean_ancillae[0])),
            cirq.Moment(cirq.X.on(n_register[0]).controlled_by(m_register[0])),
        ]

    if recursion_level == 0:
        gates.append(
            cirq.Moment(
                cirq.X.on(clean_ancillae[0]).controlled_by(n_register[0], m_register[0])
            )
        )
        gates += _quantum_addtion(
            n_register[1:],
            m_register[1:],
            clean_ancillae,
            recursion_level=recursion_level + 1,
        )
    else:
        gates += _qadd_helper_left(
            n_register[0], m_register[0], clean_ancillae[0], clean_ancillae[1]
        )
        gates += _quantum_addtion(
            n_register[1:],
            m_register[1:],
            clean_ancillae[1:],
            recursion_level=recursion_level + 1,
        )

    if recursion_level == 0:
        gates.append(
            cirq.Moment(
                cirq.X.on(clean_ancillae[0]).controlled_by(n_register[0], m_register[0])
            )
        )
        gates.append(cirq.Moment(cirq.X.on(n_register[0]).controlled_by(m_register[0])))
    else:
        gates += _qadd_helper_right(
            n_register[0], m_register[0], clean_ancillae[0], clean_ancillae[1]
        )

    return gates


def add_classical_value_gate_efficient(
    register, classical_value, clean_ancillae, ctrls=([], [])
):
    gates = []

    modded_value = classical_value % (1 << len(register))
    if modded_value == 0:
        return gates

    # if classical_value < 0:
    #     gates.append(cirq.Moment(cirq.X.on_each(*register)))

    bitstring = format(modded_value, f"0{2+len(register)}b")[2:][::-1]
    p_val = 0
    while (bitstring[p_val] != "1") and (p_val != len(bitstring)):
        p_val += 1

    if p_val == len(register):
        return gates

    p_val = min(p_val, len(register) - 1)

    reduced_classical_value = int(bitstring[p_val:][::-1], 2)
    reduced_register = register[::-1][p_val:][::-1]

    classical_value_register = clean_ancillae[: len(reduced_register)]

    gates += _load_m(reduced_classical_value, classical_value_register, ctrls=ctrls)
    gates += _quantum_addtion(
        reduced_register[::-1],
        classical_value_register[::-1],
        clean_ancillae[len(classical_value_register) :],
        recursion_level=0,
    )
    gates += _load_m(reduced_classical_value, classical_value_register, ctrls=ctrls)

    # if classical_value < 0:
    #     gates.append(cirq.Moment(cirq.X.on_each(*register)))

    return gates
