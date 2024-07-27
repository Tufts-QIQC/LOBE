import cirq


# TODO: Add uncompute flag to leave clean ancillae dirty
def is_less_than(
    qubits,
    reference,
    qbool,
    clean_ancillae,
    uncompute=True,
    ctrls=([], []),
):
    """Get a list of gates that will compute the comparator n < b for any integer b.

    The 'qubits' are assumed to store an integer ordered such that the most-significant bit
        is at position 0 and the least significant bit is at position -1. The qbool will then
        be |1> iff the integer state of qubits (n) is less than the classical reference (b).

    Args:
        qubits (List[cirq.LineQubit]): The qubit register storing an integer of which we are
            comparing against
        reference (int): The classical reference value to compare against
        qbool (cirq.LineQubit): The ancilla qubit to store the comparator value in. Will be
            |1> iff |qubits> < |b>. This qubit is assumed to be in the state |0> on input.
        clean_ancillae (List[cirq.LineQubit]): A list of qubits that are assumed to be "clean"
            (all initialized to zero).
        uncompute (bool): A classical boolean determining whether or not to clean up the used
            ancillae. Defaults to True
        ctrls (Tuple(List[cirq.LineQubit], List[int])): A set of qubits and integers that correspond to
            the control qubits and values.

    Returns:
        List[cirq.Moment]: The list of circuit operators to perform
        cirq.LineQubit: The quantum boolean storing the comparator "is less than"
        int: The number of ancillae that were used and left dirty
    """
    if not uncompute:
        raise RuntimeError(
            "is_less_than is currently only implemented for uncompute=True"
        )

    gates = []

    if reference == 1:
        gates.append(
            cirq.Moment(
                cirq.X.on(qbool).controlled_by(
                    *qubits, *ctrls[0], control_values=[0] * len(qubits) + ctrls[1]
                )
            )
        )
        gates += gates[:-1][::-1]
        return gates, qbool, 0

    elif reference == 2:
        gates.append(
            cirq.Moment(
                cirq.X.on(qbool).controlled_by(
                    *qubits[:-1],
                    *ctrls[0],
                    control_values=[0] * (len(qubits) - 1) + ctrls[1],
                )
            )
        )
        gates += gates[:-1][::-1]
        return gates, qbool, 0

    elif reference == 3:
        gates.append(
            cirq.Moment(
                cirq.X.on(clean_ancillae[0]).controlled_by(
                    *qubits[:-2],
                    *ctrls[0],
                    control_values=[0] * (len(qubits) - 2) + ctrls[1],
                )
            )
        )
        gates.append(
            cirq.Moment(
                cirq.X.on(clean_ancillae[1]).controlled_by(
                    *qubits[-2:], *ctrls[0], control_values=[1] * (2) + ctrls[1]
                )
            )
        )
        gates.append(
            cirq.Moment(
                cirq.X.on(qbool).controlled_by(
                    clean_ancillae[0],
                    clean_ancillae[1],
                    *ctrls[0],
                    control_values=[1, 0] + ctrls[1],
                )
            )
        )
        gates += gates[:-1][::-1]
        return gates, qbool, 0  # if not uncompute then used_ancillae = 2

    elif reference == 4:
        gates.append(
            cirq.Moment(
                cirq.X.on(qbool).controlled_by(
                    *qubits[:-2],
                    *ctrls[0],
                    control_values=[0] * (len(qubits) - 2) + ctrls[1],
                )
            )
        )
        return gates, qbool, 0

    elif reference == 5:
        ancillae_counter = 0
        ctrls = ([], [])
        if len(qubits) > 3:
            gates.append(
                cirq.Moment(
                    cirq.X.on(clean_ancillae[ancillae_counter]).controlled_by(
                        *qubits[:-3],
                        *ctrls[0],
                        control_values=[0] * (len(qubits) - 3) + ctrls[1],
                    )
                )
            )  # is less than 8
            ctrls[0].append(clean_ancillae[ancillae_counter])
            ctrls[1].append(1)
            ancillae_counter += 1
        gates.append(
            cirq.Moment(
                cirq.X.on(clean_ancillae[ancillae_counter]).controlled_by(
                    *qubits[-3:-1], *ctrls[0], control_values=[1, 1] + ctrls[1]
                )
            )
        )  # is equal to 6 or 7
        gates.append(
            cirq.Moment(
                cirq.X.on(clean_ancillae[ancillae_counter]).controlled_by(
                    *ctrls[0], control_values=ctrls[1]
                )
            )
        )
        ctrls[0].append(clean_ancillae[ancillae_counter])
        ctrls[1].append(1)
        ancillae_counter += 1
        gates.append(
            cirq.Moment(
                cirq.X.on(clean_ancillae[ancillae_counter]).controlled_by(
                    *qubits[-3:], *ctrls[0], control_values=[1, 0, 1] + ctrls[1]
                )
            )
        )  # is equal to 5
        gates.append(
            cirq.Moment(
                cirq.X.on(clean_ancillae[ancillae_counter]).controlled_by(
                    *ctrls[0], control_values=ctrls[1]
                )
            )
        )
        ctrls[0].append(clean_ancillae[ancillae_counter])
        ctrls[1].append(1)
        ancillae_counter += 1
        gates.append(
            cirq.Moment(
                cirq.X.on(qbool).controlled_by(
                    *ctrls[0],
                    control_values=ctrls[1],
                )
            )
        )  # is less than 8, but not 5, 6 or 7
        gates += gates[:-1][::-1]
        return gates, qbool, 0  # if not uncompute then used_ancillae = ancillae_counter

    elif reference == 6:
        ancillae_counter = 0
        ctrls = ([], [])
        if len(qubits) > 3:
            gates.append(
                cirq.Moment(
                    cirq.X.on(clean_ancillae[ancillae_counter]).controlled_by(
                        *qubits[:-3],
                        *ctrls[0],
                        control_values=[0] * (len(qubits) - 3) + ctrls[1],
                    )
                )
            )  # is less than 8
            ctrls[0].append(clean_ancillae[ancillae_counter])
            ctrls[1].append(1)
            ancillae_counter += 1
        gates.append(
            cirq.Moment(
                cirq.X.on(clean_ancillae[ancillae_counter]).controlled_by(
                    *qubits[-3:-1], *ctrls[0], control_values=[1, 1] + ctrls[1]
                )
            )
        )  # is equal to 6 or 7
        gates.append(
            cirq.Moment(
                cirq.X.on(clean_ancillae[ancillae_counter]).controlled_by(
                    *ctrls[0], control_values=ctrls[1]
                )
            )
        )
        ctrls[0].append(clean_ancillae[ancillae_counter])
        ctrls[1].append(1)
        ancillae_counter += 1

        gates.append(
            cirq.Moment(
                cirq.X.on(qbool).controlled_by(
                    *ctrls[0],
                    control_values=ctrls[1],
                )
            )
        )  # is less than 8, but not 6 or 7
        gates += gates[:-1][::-1]
        return gates, qbool, 0  # if not uncompute then used_ancillae = ancillae_counter

    elif reference == 7:
        ancillae_counter = 0
        ctrls = ([], [])
        if len(qubits) > 3:
            gates.append(
                cirq.Moment(
                    cirq.X.on(clean_ancillae[ancillae_counter]).controlled_by(
                        *qubits[:-3],
                        *ctrls[0],
                        control_values=[0] * (len(qubits) - 3) + ctrls[1],
                    )
                )
            )  # is less than 8
            ctrls[0].append(clean_ancillae[ancillae_counter])
            ctrls[1].append(1)
            ancillae_counter += 1
        gates.append(
            cirq.Moment(
                cirq.X.on(clean_ancillae[ancillae_counter]).controlled_by(
                    *qubits[-3:], *ctrls[0], control_values=[1, 1, 1] + ctrls[1]
                )
            )
        )  # is equal to 7
        gates.append(
            cirq.Moment(
                cirq.X.on(clean_ancillae[ancillae_counter]).controlled_by(
                    *ctrls[0], control_values=ctrls[1]
                )
            )
        )
        ctrls[0].append(clean_ancillae[ancillae_counter])
        ctrls[1].append(1)
        ancillae_counter += 1

        gates.append(
            cirq.Moment(
                cirq.X.on(qbool).controlled_by(
                    *ctrls[0],
                    control_values=ctrls[1],
                )
            )
        )  # is less than 8, but not 7
        gates += gates[:-1][::-1]
        return gates, qbool, 0  # if not uncompute then used_ancillae = ancillae_counter

    elif reference == 8:
        gates.append(
            cirq.Moment(
                cirq.X.on(qbool).controlled_by(
                    *qubits[:-3],
                    *ctrls[0],
                    control_values=[0] * (len(qubits) - 3) + ctrls[1],
                )
            )
        )
        return gates, qbool, 0

    elif reference == 9:
        ancillae_counter = 0
        ctrls = ([], [])
        if len(qubits) > 4:
            gates.append(
                cirq.Moment(
                    cirq.X.on(clean_ancillae[ancillae_counter]).controlled_by(
                        *qubits[:-4],
                        *ctrls[0],
                        control_values=[0] * (len(qubits) - 4) + ctrls[1],
                    )
                )
            )  # is less than 16
            ctrls[0].append(clean_ancillae[ancillae_counter])
            ctrls[1].append(1)
            ancillae_counter += 1

        gates.append(
            cirq.Moment(
                cirq.X.on(clean_ancillae[ancillae_counter]).controlled_by(
                    *qubits[-4:-2], *ctrls[0], control_values=[1, 1] + ctrls[1]
                )
            )
        )  # is equal to 15, 14, 13, or 12
        ctrls[0].append(clean_ancillae[ancillae_counter])
        ctrls[1].append(0)
        ancillae_counter += 1

        gates.append(
            cirq.Moment(
                cirq.X.on(clean_ancillae[ancillae_counter]).controlled_by(
                    *qubits[-4:-1], *ctrls[0], control_values=[1, 0, 1] + ctrls[1]
                )
            )
        )  # is equal to 11 or 10
        ctrls[0].append(clean_ancillae[ancillae_counter])
        ctrls[1].append(0)
        ancillae_counter += 1

        gates.append(
            cirq.Moment(
                cirq.X.on(clean_ancillae[ancillae_counter]).controlled_by(
                    *qubits[-4:], *ctrls[0], control_values=[1, 0, 0, 1] + ctrls[1]
                )
            )
        )  # is equal to 9
        ctrls[0].append(clean_ancillae[ancillae_counter])
        ctrls[1].append(0)
        ancillae_counter += 1

        gates.append(
            cirq.Moment(
                cirq.X.on(qbool).controlled_by(
                    *ctrls[0],
                    control_values=ctrls[1],
                )
            )
        )  # is less than 16, but not 15, 14, 13, 12, 11, 10, or 9
        gates += gates[:-1][::-1]
        return gates, qbool, 0  # if not uncompute then used_ancillae = ancillae_counter

    elif reference == 10:
        ancillae_counter = 0
        ctrls = ([], [])
        if len(qubits) > 4:
            gates.append(
                cirq.Moment(
                    cirq.X.on(clean_ancillae[ancillae_counter]).controlled_by(
                        *qubits[:-4],
                        *ctrls[0],
                        control_values=[0] * (len(qubits) - 4) + ctrls[1],
                    )
                )
            )  # is less than 16
            ctrls[0].append(clean_ancillae[ancillae_counter])
            ctrls[1].append(1)
            ancillae_counter += 1

        gates.append(
            cirq.Moment(
                cirq.X.on(clean_ancillae[ancillae_counter]).controlled_by(
                    *qubits[-4:-2], *ctrls[0], control_values=[1, 1] + ctrls[1]
                )
            )
        )  # is equal to 15, 14, 13, or 12
        ctrls[0].append(clean_ancillae[ancillae_counter])
        ctrls[1].append(0)
        ancillae_counter += 1

        gates.append(
            cirq.Moment(
                cirq.X.on(clean_ancillae[ancillae_counter]).controlled_by(
                    *qubits[-4:-1], *ctrls[0], control_values=[1, 0, 1] + ctrls[1]
                )
            )
        )  # is equal to 11 or 10
        ctrls[0].append(clean_ancillae[ancillae_counter])
        ctrls[1].append(0)
        ancillae_counter += 1

        gates.append(
            cirq.Moment(
                cirq.X.on(qbool).controlled_by(
                    *ctrls[0],
                    control_values=ctrls[1],
                )
            )
        )  # is less than 16, but not 15, 14, 13, 12, 11, or 10
        gates += gates[:-1][::-1]
        return gates, qbool, 0  # if not uncompute then used_ancillae = ancillae_counter

    elif reference == 11:
        ancillae_counter = 0
        ctrls = ([], [])
        if len(qubits) > 4:
            gates.append(
                cirq.Moment(
                    cirq.X.on(clean_ancillae[ancillae_counter]).controlled_by(
                        *qubits[:-4],
                        *ctrls[0],
                        control_values=[0] * (len(qubits) - 4) + ctrls[1],
                    )
                )
            )  # is less than 16
            ctrls[0].append(clean_ancillae[ancillae_counter])
            ctrls[1].append(1)
            ancillae_counter += 1

        gates.append(
            cirq.Moment(
                cirq.X.on(clean_ancillae[ancillae_counter]).controlled_by(
                    *qubits[-4:-2], *ctrls[0], control_values=[1, 1] + ctrls[1]
                )
            )
        )  # is equal to 15, 14, 13, or 12
        ctrls[0].append(clean_ancillae[ancillae_counter])
        ctrls[1].append(0)
        ancillae_counter += 1

        gates.append(
            cirq.Moment(
                cirq.X.on(clean_ancillae[ancillae_counter]).controlled_by(
                    *qubits[-4:], *ctrls[0], control_values=[1, 0, 1, 1] + ctrls[1]
                )
            )
        )  # is equal to 11
        ctrls[0].append(clean_ancillae[ancillae_counter])
        ctrls[1].append(0)
        ancillae_counter += 1

        gates.append(
            cirq.Moment(
                cirq.X.on(qbool).controlled_by(
                    *ctrls[0],
                    control_values=ctrls[1],
                )
            )
        )  # is less than 16, but not 15, 14, 13, 12, or 11
        gates += gates[:-1][::-1]
        return gates, qbool, 0  # if not uncompute then used_ancillae = ancillae_counter

    elif reference == 12:
        ancillae_counter = 0
        ctrls = ([], [])
        if len(qubits) > 4:
            gates.append(
                cirq.Moment(
                    cirq.X.on(clean_ancillae[ancillae_counter]).controlled_by(
                        *qubits[:-4],
                        *ctrls[0],
                        control_values=[0] * (len(qubits) - 4) + ctrls[1],
                    )
                )
            )  # is less than 16
            ctrls[0].append(clean_ancillae[ancillae_counter])
            ctrls[1].append(1)
            ancillae_counter += 1

        gates.append(
            cirq.Moment(
                cirq.X.on(clean_ancillae[ancillae_counter]).controlled_by(
                    *qubits[-4:-2], *ctrls[0], control_values=[1, 1] + ctrls[1]
                )
            )
        )  # is equal to 15, 14, 13, or 12
        ctrls[0].append(clean_ancillae[ancillae_counter])
        ctrls[1].append(0)
        ancillae_counter += 1

        gates.append(
            cirq.Moment(
                cirq.X.on(qbool).controlled_by(
                    *ctrls[0],
                    control_values=ctrls[1],
                )
            )
        )  # is less than 16, but not 15, 14, 13, or 12
        gates += gates[:-1][::-1]
        return gates, qbool, 0  # if not uncompute then used_ancillae = ancillae_counter

    elif reference == 13:
        ancillae_counter = 0
        ctrls = ([], [])
        if len(qubits) > 4:
            gates.append(
                cirq.Moment(
                    cirq.X.on(clean_ancillae[ancillae_counter]).controlled_by(
                        *qubits[:-4],
                        *ctrls[0],
                        control_values=[0] * (len(qubits) - 4) + ctrls[1],
                    )
                )
            )  # is less than 16
            ctrls[0].append(clean_ancillae[ancillae_counter])
            ctrls[1].append(1)
            ancillae_counter += 1

        gates.append(
            cirq.Moment(
                cirq.X.on(clean_ancillae[ancillae_counter]).controlled_by(
                    *qubits[-4:-1], *ctrls[0], control_values=[1, 1, 1] + ctrls[1]
                )
            )
        )  # is equal to 15 or 14
        ctrls[0].append(clean_ancillae[ancillae_counter])
        ctrls[1].append(0)
        ancillae_counter += 1

        gates.append(
            cirq.Moment(
                cirq.X.on(clean_ancillae[ancillae_counter]).controlled_by(
                    *qubits[-4:], *ctrls[0], control_values=[1, 1, 0, 1] + ctrls[1]
                )
            )
        )  # is equal to 13
        ctrls[0].append(clean_ancillae[ancillae_counter])
        ctrls[1].append(0)
        ancillae_counter += 1

        gates.append(
            cirq.Moment(
                cirq.X.on(qbool).controlled_by(
                    *ctrls[0],
                    control_values=ctrls[1],
                )
            )
        )  # is less than 16, but not 15, 14 or 13
        gates += gates[:-1][::-1]
        return gates, qbool, 0  # if not uncompute then used_ancillae = ancillae_counter

    elif reference == 14:
        ancillae_counter = 0
        ctrls = ([], [])
        if len(qubits) > 4:
            gates.append(
                cirq.Moment(
                    cirq.X.on(clean_ancillae[ancillae_counter]).controlled_by(
                        *qubits[:-4],
                        *ctrls[0],
                        control_values=[0] * (len(qubits) - 4) + ctrls[1],
                    )
                )
            )  # is less than 16
            ctrls[0].append(clean_ancillae[ancillae_counter])
            ctrls[1].append(1)
            ancillae_counter += 1

        gates.append(
            cirq.Moment(
                cirq.X.on(clean_ancillae[ancillae_counter]).controlled_by(
                    *qubits[-4:-1], *ctrls[0], control_values=[1, 1, 1] + ctrls[1]
                )
            )
        )  # is equal to 15 or 14
        ctrls[0].append(clean_ancillae[ancillae_counter])
        ctrls[1].append(0)
        ancillae_counter += 1

        gates.append(
            cirq.Moment(
                cirq.X.on(qbool).controlled_by(
                    *ctrls[0],
                    control_values=ctrls[1],
                )
            )
        )  # is less than 16, but not 15 or 14
        gates += gates[:-1][::-1]
        return gates, qbool, 0  # if not uncompute then used_ancillae = ancillae_counter

    elif reference == 15:
        ancillae_counter = 0
        ctrls = ([], [])
        if len(qubits) > 4:
            gates.append(
                cirq.Moment(
                    cirq.X.on(clean_ancillae[ancillae_counter]).controlled_by(
                        *qubits[:-4],
                        *ctrls[0],
                        control_values=[0] * (len(qubits) - 4) + ctrls[1],
                    )
                )
            )  # is less than 16
            ctrls[0].append(clean_ancillae[ancillae_counter])
            ctrls[1].append(1)
            ancillae_counter += 1

        gates.append(
            cirq.Moment(
                cirq.X.on(clean_ancillae[ancillae_counter]).controlled_by(
                    *qubits[-4:], *ctrls[0], control_values=[1, 1, 1, 1] + ctrls[1]
                )
            )
        )  # is equal to 15
        ctrls[0].append(clean_ancillae[ancillae_counter])
        ctrls[1].append(0)
        ancillae_counter += 1

        gates.append(
            cirq.Moment(
                cirq.X.on(qbool).controlled_by(
                    *ctrls[0],
                    control_values=ctrls[1],
                )
            )
        )  # is less than 16, but not 15
        gates += gates[:-1][::-1]
        return gates, qbool, 0  # if not uncompute then used_ancillae = ancillae_counter

    raise RuntimeError("is_less_than only supports 0 < reference < 16")


def is_greater_than(
    qubits,
    reference,
    qbool,
    clean_ancillae,
    uncompute=True,
    ctrls=([], []),
):
    """Get a list of gates that will compute the comparator n > b for any integer b.

    The 'qubits' are assumed to store an integer ordered such that the most-significant bit
        is at position 0 and the least significant bit is at position -1. The qbool will then
        be |1> iff the integer state of qubits (n) is greater than the classical reference (b).

    Args:
        qubits (List[cirq.LineQubit]): The qubit register storing an integer of which we are
            comparing against
        reference (int): The classical reference value to compare against
        qbool (cirq.LineQubit): The ancilla qubit to store the comparator value in. Will be
            |1> iff |qubits> < |b>. This qubit is assumed to be in the state |0> on input.
        clean_ancillae (List[cirq.LineQubit]): A list of qubits that are assumed to be "clean"
            (all initialized to zero).
        uncompute (bool): A classical boolean determining whether or not to clean up the used
            ancillae. Defaults to True
        ctrls (Tuple(List[cirq.LineQubit], List[int])): A set of qubits and integers that correspond to
            the control qubits and values.

    Returns:
        List[cirq.Moment]: The list of circuit operators to perform
        cirq.LineQubit: The quantum boolean storing the comparator "is less than"
        int: The number of ancillae that were used and left dirty
    """
    assert reference < ((1 << len(qubits)) - 1)
    assert reference > (((1 << len(qubits)) - 1) - 15)
    gates = []
    gates.append(cirq.Moment(cirq.X.on_each(*qubits)))
    reference = (1 << len(qubits)) - reference - 1
    additional_gates, qbool, used_ancillae = is_less_than(
        qubits, reference, qbool, clean_ancillae, uncompute=uncompute, ctrls=ctrls
    )
    gates += additional_gates
    gates.append(cirq.Moment(cirq.X.on_each(*qubits)))
    return gates, qbool, used_ancillae
