import cirq
import numpy as np
from ._grover_rudolph import _grover_rudolph


def get_target_state(coefficients):
    number_of_qubits = max(int(np.ceil(np.log2(len(coefficients)))), 1)
    norm = sum(np.abs(coefficients))
    target_state = np.zeros(1 << number_of_qubits, dtype=complex)
    for i, coeff in enumerate(coefficients):
        bitstring = format(i, f"0{2+number_of_qubits}b")[2:]
        target_state[int(bitstring, 2)] = np.sqrt(np.abs(coeff) / norm)
    return target_state


def get_grover_rudolph_instructions(target_state):
    """Helper function to parse the instructions returned by the grover-rudolph package.

    Args:
        target_state (np.ndarray): The target state to prepare

    Returns:
        List[List[Tuple[int, List[int], float, float]]]: A list of the rotations to perform
    """
    num_qubits = int(np.ceil(np.log2(len(target_state))))
    reordered_target_state = np.zeros(1 << num_qubits, dtype=complex)
    for i, coeff in enumerate(target_state):
        bitstring = format(i, f"0{2+num_qubits}b")[2:]
        reordered_target_state[int(bitstring[::-1], 2)] = coeff

    gate_list = _grover_rudolph(target_state)

    parsed_instructions = []

    # index of list is qubits that gates act on; item in list is a dictionary with gate instructions
    for qubit_index, instructions in enumerate(gate_list):

        # keys of instructions are the computational basis states to control the previous qubits on;
        #   value is a Tuple with the angle of ry gate followed by the angle of the phase gate in radians
        instructions = list(instructions.items())

        for controls, (ry_angle, rz_angle) in instructions:
            control_values = [int(ctrl) for ctrl in controls]

            while (len(control_values) + 1) > len(parsed_instructions):
                parsed_instructions.append([])

            parsed_instructions[len(control_values)].append(
                (qubit_index, control_values, ry_angle, rz_angle)
            )

    return parsed_instructions


def get_multiplexed_grover_rudolph_circuit(
    instructions,
    qubits,
    clean_ancillae,
    previous_control_values=[],
    ctrls=([], []),
    dagger=False,
    count_numerics=False,
):
    """Generate the gate list to perform the "multiplexed" grover-rudolph circuit.

    Args:
        instructions (List[List[Tuple[int, List[int], float, float]]]): The list of rotation gates to perform
        qubits (List[cirq.LineQubit]): The qubits on which to prepare the target state
        clean_ancillae (List[cirq.LineQubit]): Ancillae qubits promised to begin and end in the zero state
        previous_control_values (List[int]): A list of previously assumed quantum controls on the stack when recursing
        ctrls (Tuple[List[cirq.LineQubit], List[int]]): The quantum controls to use on subsequent operations
        dagger (bool): Determines if we want the daggered version of the circuit
        count_numerics (bool): Determines if we want to count the number of gates used

    Returns:
        List[cirq.Moment]: A list of the gate operations to perform in the quantum circuit
        Optional[Dict]: A dictionary storing the numerical gate count estimates
    """
    gate_numerics = {
        "num_rotations": 0,
        "num_left_elbows": 0,
        "num_right_elbows": 0,
    }
    gates = []

    current_qubit_index = len(previous_control_values)

    if (
        (len(instructions) == 0)
        or (current_qubit_index > len(qubits) - 1)
        or (len(instructions[current_qubit_index]) == 0)
    ):
        if count_numerics:
            return gates, gate_numerics
        else:
            return gates

    # Case 1: Rotation with current controls
    for qubit_index, control_values, ry_angle, rz_angle in instructions[
        current_qubit_index
    ]:
        if dagger:
            ry_angle *= -1
            rz_angle *= -1
        if control_values == previous_control_values:
            gate_numerics["num_rotations"] += 1
            gate = (
                cirq.ry(ry_angle)
                .on(qubits[qubit_index])
                .controlled_by(*ctrls[0], control_values=ctrls[1])
            )
            if dagger:
                gates = [gate] + gates
            else:
                gates.append(gate)

            if not np.isclose(rz_angle, 0):
                gate_numerics["num_rotations"] += 1
                gates = (
                    cirq.ZPowGate(exponent=rz_angle / np.pi, global_shift=0)
                    .on(qubits[qubit_index])
                    .controlled_by(*ctrls[0], control_values=ctrls[1])
                )

                if dagger:
                    gates = [gate] + gates
                else:
                    gates.append(gate)

    # Case 2: add control & recurse
    if len(previous_control_values) > 0:
        return_vals = get_multiplexed_grover_rudolph_circuit(
            instructions,
            qubits,
            clean_ancillae=clean_ancillae[1:],
            previous_control_values=previous_control_values + [0],
            ctrls=([clean_ancillae[0]], [1]),
            dagger=dagger,
        )
        if count_numerics:
            gates_to_add = return_vals[0]
            gate_numerics_to_add = return_vals[1]
        else:
            gates_to_add = return_vals
        if len(gates_to_add) > 0:
            # add left-elbow
            left_elbow = cirq.X.on(clean_ancillae[0]).controlled_by(
                *ctrls[0], qubits[current_qubit_index], control_values=ctrls[1] + [0]
            )
            right_elbow = left_elbow
            gates_to_add = [left_elbow] + gates_to_add + [right_elbow]
            if count_numerics:
                gate_numerics["num_rotations"] += gate_numerics_to_add["num_rotations"]
                gate_numerics["num_left_elbows"] += (
                    gate_numerics_to_add["num_left_elbows"] + 1
                )
                gate_numerics["num_right_elbows"] += (
                    gate_numerics_to_add["num_right_elbows"] + 1
                )

        if dagger:
            gates = [gates_to_add] + gates
        else:
            gates += gates_to_add

        return_vals = get_multiplexed_grover_rudolph_circuit(
            instructions,
            qubits,
            clean_ancillae=clean_ancillae[1:],
            previous_control_values=previous_control_values + [1],
            ctrls=([clean_ancillae[0]], [1]),
            dagger=dagger,
        )
        if count_numerics:
            gates_to_add = return_vals[0]
            gate_numerics_to_add = return_vals[1]
        else:
            gates_to_add = return_vals
        if len(gates_to_add) > 0:
            # add left-elbow
            left_elbow = cirq.X.on(clean_ancillae[0]).controlled_by(
                *ctrls[0], qubits[current_qubit_index], control_values=ctrls[1] + [1]
            )
            right_elbow = left_elbow
            gates_to_add = [left_elbow] + gates_to_add + [right_elbow]
            if count_numerics:
                gate_numerics["num_rotations"] += gate_numerics_to_add["num_rotations"]
                gate_numerics["num_left_elbows"] += (
                    gate_numerics_to_add["num_left_elbows"] + 1
                )
                gate_numerics["num_right_elbows"] += (
                    gate_numerics_to_add["num_right_elbows"] + 1
                )

        if dagger:
            gates = [gates_to_add] + gates
        else:
            gates += gates_to_add
    else:
        return_vals = get_multiplexed_grover_rudolph_circuit(
            instructions,
            qubits,
            clean_ancillae=clean_ancillae,
            previous_control_values=previous_control_values + [0],
            ctrls=(
                qubits[: len(previous_control_values) + 1],
                previous_control_values + [0],
            ),
            dagger=dagger,
        )
        if count_numerics:
            gates_to_add = return_vals[0]
            gate_numerics_to_add = return_vals[1]
        else:
            gates_to_add = return_vals
        if dagger:
            gates = [gates_to_add] + gates
        else:
            gates += gates_to_add

        if count_numerics:
            gate_numerics["num_rotations"] += gate_numerics_to_add["num_rotations"]
            gate_numerics["num_left_elbows"] += gate_numerics_to_add["num_left_elbows"]
            gate_numerics["num_right_elbows"] += gate_numerics_to_add[
                "num_right_elbows"
            ]

        return_vals = get_multiplexed_grover_rudolph_circuit(
            instructions,
            qubits,
            clean_ancillae=clean_ancillae,
            previous_control_values=previous_control_values + [1],
            ctrls=(
                qubits[: len(previous_control_values) + 1],
                previous_control_values + [1],
            ),
            dagger=dagger,
        )
        if count_numerics:
            gates_to_add = return_vals[0]
            gate_numerics_to_add = return_vals[1]
        else:
            gates_to_add = return_vals
        if dagger:
            gates = [gates_to_add] + gates
        else:
            gates += gates_to_add

        if count_numerics:
            gate_numerics["num_rotations"] += gate_numerics_to_add["num_rotations"]
            gate_numerics["num_left_elbows"] += gate_numerics_to_add["num_left_elbows"]
            gate_numerics["num_right_elbows"] += gate_numerics_to_add[
                "num_right_elbows"
            ]

    if count_numerics:
        return gates, gate_numerics
    else:
        return gates


def add_prepare_circuit(qubits, target_state, dagger=False):
    """Add a quantum circuit that prepares the target state (arbitrary quantum state) when acting on the all-zero state.

    Implementation based on: https://arxiv.org/abs/quant-ph/0208112

    Args:
        circuit (cirq.Circuit): The quantum circuit to add instructions to
        qubits (List[cirq.LineQubit]): The qubits that the circuit acts upon
        target_state (np.ndarray): A numpy array describing the arbitrary quantum state

    Returns:
        cirq.Circuit: The quantum circuit including the prepare oracle
    """
    reordered_target_state = np.zeros(1 << len(qubits))
    for i, coeff in enumerate(target_state):
        bitstring = format(i, f"0{2+len(qubits)}b")[2:]
        reordered_target_state[int(bitstring[::-1], 2)] = coeff

    gates = []
    gate_list = _grover_rudolph(target_state)

    if dagger:
        gate_list = gate_list[::-1]

    # index of list is qubits that gates act on; item in list is a dictionary with gate instructions
    for qubit_index, instructions in enumerate(gate_list):

        if dagger:
            qubit_index = len(gate_list) - qubit_index - 1

        # keys of instructions are the computational basis states to control the previous qubits on;
        #   value is a Tuple with the angle of ry gate followed by the angle of the phase gate in radians
        instructions = list(instructions.items())
        if dagger:
            instructions = instructions[::-1]
        for controls, (ry_angle, rz_angle) in instructions:

            if dagger:
                ry_angle *= -1
                rz_angle *= -1

            # map string controls to integers
            control_values = [int(ctrl) for ctrl in controls]

            # create phase gate with appropriate angle
            phase_gate = cirq.ZPowGate(exponent=rz_angle / np.pi, global_shift=0)

            if not dagger:
                # Controlled-Ry
                gates.append(
                    cirq.ry(ry_angle)
                    .on(qubits[qubit_index])
                    .controlled_by(*qubits[:qubit_index], control_values=control_values)
                )

            if not np.isclose(rz_angle, 0):
                # Controlled-Phase
                gates.append(
                    phase_gate.on(qubits[qubit_index]).controlled_by(
                        *qubits[:qubit_index], control_values=control_values
                    )
                )

            if dagger:
                # Controlled-Ry
                gates.append(
                    cirq.ry(ry_angle)
                    .on(qubits[qubit_index])
                    .controlled_by(*qubits[:qubit_index], control_values=control_values)
                )

    return gates
