import cirq
import numpy as np
from .metrics import CircuitMetrics


def add_classical_value(register, classical_value, clean_ancillae, ctrls=([], [])):
    incrementers_circuit, incrementers_metrics = add_classical_value_incrementers(
        register, classical_value, clean_ancillae, ctrls=ctrls
    )
    gate_efficient_circuit, gate_efficient_metrics = add_classical_value_incrementers(
        register, classical_value, clean_ancillae, ctrls=ctrls
    )

    if gate_efficient_metrics.number_of_elbows < incrementers_metrics.number_of_elbows:
        return gate_efficient_circuit, gate_efficient_metrics
    else:
        return incrementers_circuit, incrementers_metrics


def add_classical_value_incrementers(
    register, classical_value, clean_ancillae, ctrls=([], [])
):
    classical_value = classical_value % (1 << len(register))
    adder_metrics = CircuitMetrics()

    if classical_value == 0:
        return [], adder_metrics

    options = [(classical_value, False), ((1 << len(register)) - classical_value, True)]

    outcomes = []
    metrics_list = []
    for value, flip_decr in options:
        decrement = False
        if value < 0:
            decrement = True
        if flip_decr:
            decrement = not decrement

        num_levels = int(np.ceil(np.log2(np.abs(value)))) + 1
        gates = []
        value = np.abs(value)
        option_metrics = CircuitMetrics()
        for level in range(num_levels - 1, -1, -1):

            qubits_involved = register
            if level > 0:
                qubits_involved = register[:-level]

            if (1 << level) <= value:
                _gates, _metrics = add_incrementer(
                    qubits_involved,
                    clean_ancillae[: len(qubits_involved) - 1],
                    decrement=decrement,
                    ctrls=ctrls,
                )
                gates += _gates
                option_metrics += _metrics
                value -= 1 << level
        outcomes.append(gates)
        metrics_list.append(option_metrics)

    if metrics_list[1].number_of_elbows < metrics_list[0].number_of_elbows:
        return outcomes[1], metrics_list[1]
    return outcomes[0], metrics_list[0]


def add_incrementer(register, clean_ancilla, decrement=False, ctrls=([], [])):
    """This function adds an oracle to our circuit that increments the binary value of the register by +/- 1.

    Implementation: https://algassert.com/circuits/2015/06/12/Constructing-Large-Increment-Gates.html
    """
    assert len(ctrls[0]) <= 1
    gates = []
    incrementer_metrics = CircuitMetrics()

    if decrement:
        gates.append(cirq.Moment(cirq.X.on_each(*register)))
        _gates, _metrics = add_incrementer(
            register, clean_ancilla, decrement=False, ctrls=ctrls
        )
        gates += _gates
        incrementer_metrics += _metrics
        gates.append(cirq.Moment(cirq.X.on_each(*register)))
        return gates, incrementer_metrics

    if len(ctrls[0]) == 1:
        if ctrls[1][0] == 0:
            gates.append(cirq.X.on(ctrls[0][0]))
        _gates, _metrics = add_incrementer(register + ctrls[0], clean_ancilla)
        gates += _gates
        incrementer_metrics += _metrics
        gates.append(cirq.Moment(cirq.X.on(ctrls[0][0])))
        if ctrls[1][0] == 0:
            gates.append(cirq.X.on(ctrls[0][0]))
        return gates, incrementer_metrics

    if len(register) == 1:
        gates.append(cirq.Moment(cirq.X.on(register[0])))
        return gates, incrementer_metrics

    if len(register) == 2:
        gates.append(
            cirq.Moment(
                cirq.X.on(register[0]).controlled_by(register[1], control_values=[1])
            )
        )
        gates.append(cirq.Moment(cirq.X.on(register[1])))
        return gates, incrementer_metrics
    else:
        register = register[::-1]
        counter = 0
        first_control = register[0]
        second_control = register[1]
        while counter < len(register) - 2:
            clean = clean_ancilla[counter]
            _gates, _metrics = _incremeneter_helper_left(
                first_control, second_control, clean
            )
            gates += _gates
            incrementer_metrics += _metrics
            first_control = register[counter + 2]
            second_control = clean_ancilla[counter]
            counter += 1
        counter -= 1
        while counter >= 1:
            first_control = register[counter + 1]
            second_control = clean_ancilla[counter - 1]
            clean = clean_ancilla[counter]
            target = register[counter + 2]
            _gates, _metrics = _incremeneter_helper_right(
                first_control,
                second_control,
                clean,
                target,
            )
            gates += _gates
            incrementer_metrics += _metrics
            counter -= 1

        _gates, _metrics = _incremeneter_helper_right(
            register[0],
            register[1],
            clean_ancilla[0],
            register[2],
        )
        gates += _gates
        incrementer_metrics += _metrics
        gates.append(cirq.Moment(cirq.X.on(register[1]).controlled_by(register[0])))
        gates.append(cirq.Moment(cirq.X.on(register[0])))

    return gates, incrementer_metrics


def add_classical_value_gate_efficient(
    register, classical_value, clean_ancillae, ctrls=([], [])
):
    assert len(ctrls[0]) <= 1
    gates = []
    adder_metrics = CircuitMetrics()

    modded_value = classical_value % (1 << len(register))
    if modded_value == 0:
        return gates, adder_metrics

    bitstring = format(modded_value, f"0{2+len(register)}b")[2:][::-1]
    p_val = _get_p_val(modded_value, len(register))

    if p_val == len(register):
        return gates, adder_metrics

    p_val = min(p_val, len(register) - 1)

    reduced_classical_value = int(bitstring[p_val:][::-1], 2)
    reduced_register = register[::-1][p_val:][::-1]

    classical_value_register = clean_ancillae[: len(reduced_register)]

    adder_metrics.add_to_clean_ancillae_usage(len(classical_value_register))
    gates += _load_m(reduced_classical_value, classical_value_register, ctrls=ctrls)

    _gates, _metrics = _quantum_addtion(
        reduced_register[::-1],
        classical_value_register[::-1],
        clean_ancillae[len(classical_value_register) :],
        recursion_level=0,
    )
    gates += _gates
    adder_metrics += _metrics

    gates += _load_m(reduced_classical_value, classical_value_register, ctrls=ctrls)
    adder_metrics.add_to_clean_ancillae_usage(-len(classical_value_register))

    return gates, adder_metrics


def _incremeneter_helper_left(first_control, second_control, clean):
    _gates = []
    _metrics = CircuitMetrics()

    _metrics.add_to_clean_ancillae_usage(1)
    _metrics.number_of_elbows += 1
    _gates.append(
        cirq.Moment(cirq.X.on(clean).controlled_by(first_control, second_control))
    )

    return _gates, _metrics


def _incremeneter_helper_right(first_control, second_control, clean, target):
    _gates = []
    _metrics = CircuitMetrics()

    _gates.append(cirq.Moment(cirq.X.on(target).controlled_by(clean)))
    _gates.append(
        cirq.Moment(cirq.X.on(clean).controlled_by(first_control, second_control))
    )
    _metrics.add_to_clean_ancillae_usage(-1)

    return _gates, _metrics


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
    _metrics = CircuitMetrics()
    gates.append(cirq.Moment(cirq.X.on(m_bit).controlled_by(carry_in)))
    gates.append(cirq.Moment(cirq.X.on(n_bit).controlled_by(carry_in)))
    gates.append(cirq.Moment(cirq.X.on(carry_out).controlled_by(n_bit, m_bit)))
    gates.append(cirq.Moment(cirq.X.on(carry_out).controlled_by(carry_in)))
    _metrics.add_to_clean_ancillae_usage(1)
    _metrics.number_of_elbows += 1
    return gates, _metrics


def _qadd_helper_right(n_bit, m_bit, carry_in, carry_out):
    gates = []
    _metrics = CircuitMetrics()
    gates.append(cirq.Moment(cirq.X.on(carry_out).controlled_by(carry_in)))
    gates.append(cirq.Moment(cirq.X.on(carry_out).controlled_by(n_bit, m_bit)))
    gates.append(cirq.Moment(cirq.X.on(m_bit).controlled_by(carry_in)))
    gates.append(cirq.Moment(cirq.X.on(n_bit).controlled_by(m_bit)))
    _metrics.add_to_clean_ancillae_usage(-1)
    return gates, _metrics


def _quantum_addtion(n_register, m_register, clean_ancillae, recursion_level=0):
    gates = []
    addition_metrics = CircuitMetrics()

    if len(n_register) == 0:
        return gates, addition_metrics
    elif len(n_register) == 1:
        return [
            cirq.Moment(cirq.X.on(n_register[0]).controlled_by(clean_ancillae[0])),
            cirq.Moment(cirq.X.on(n_register[0]).controlled_by(m_register[0])),
        ], addition_metrics

    if recursion_level == 0:
        addition_metrics.add_to_clean_ancillae_usage(1)
        addition_metrics.number_of_elbows += 1

        gates.append(
            cirq.Moment(
                cirq.X.on(clean_ancillae[0]).controlled_by(n_register[0], m_register[0])
            )
        )
        _gates, _metrics = _quantum_addtion(
            n_register[1:],
            m_register[1:],
            clean_ancillae,
            recursion_level=recursion_level + 1,
        )
        gates += _gates
        addition_metrics += _metrics

        gates.append(
            cirq.Moment(
                cirq.X.on(clean_ancillae[0]).controlled_by(n_register[0], m_register[0])
            )
        )
        addition_metrics.add_to_clean_ancillae_usage(-1)
        gates.append(cirq.Moment(cirq.X.on(n_register[0]).controlled_by(m_register[0])))
    else:
        _gates, _metrics = _qadd_helper_left(
            n_register[0], m_register[0], clean_ancillae[0], clean_ancillae[1]
        )
        gates += _gates
        addition_metrics += _metrics

        _gates, _metrics = _quantum_addtion(
            n_register[1:],
            m_register[1:],
            clean_ancillae[1:],
            recursion_level=recursion_level + 1,
        )
        gates += _gates
        addition_metrics += _metrics

        _gates, _metrics = _qadd_helper_right(
            n_register[0], m_register[0], clean_ancillae[0], clean_ancillae[1]
        )
        gates += _gates
        addition_metrics += _metrics

    return gates, addition_metrics


def _get_p_val(classical_value, number_of_bits):
    bitstring = format(classical_value, f"0{2+number_of_bits}b")[2:][::-1]
    p_val = 0
    while bitstring[p_val] != "1":
        p_val += 1
        if p_val == len(bitstring):
            break
    return p_val
