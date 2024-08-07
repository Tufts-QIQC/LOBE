import cirq
import numpy as np
import scipy as sp


def get_target_state(coefficients):
    number_of_qubits = max(int(np.ceil(np.log2(len(coefficients)))), 1)
    norm = sum(np.abs(coefficients))
    target_state = np.zeros(1 << number_of_qubits)
    for i, coeff in enumerate(coefficients):
        bitstring = format(i, f"0{2+number_of_qubits}b")[2:]
        target_state[int(bitstring, 2)] = np.sqrt(np.abs(coeff) / norm)
    return target_state


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


_ZERO = 1e-8


def _grover_rudolph(vector, *, optimization=False):
    """
    Generate a list of dictonaries for the angles given the amplitude vector
    Each dictonary is of the form:
        {key = ('0' if apply controlled on the state 0, '1' if controlled on 1, 'e' if apply identy) : value = [angle, phase]
        {'00' : [1.2, 0.]} the gate is a rotation of 1.2 and a phase gate with phase 0, controlled on the state |00>

    You are basically building the cicuit vertically, where each element of the dictionary is one layer of the circuit
    if the dictonary is in position 'i' of the list (starting from 0), its key will be of length 'i', thus the controls act on the fist i qubits

    Args:
        vector: compressed version (only non-zero elements) of the sparse state vector to be prepared
        optimization: decide if optimize the angles or not, defaults to True

    Returns:
        a sequence of controlled gates to be applied.
    """
    vector = _sanitize_sparse_state_vector(vector)

    nonzero_values = vector.data
    nonzero_locations = vector.nonzero()[1]
    N_qubit = _number_of_qubits(nonzero_values)

    final_gates = []

    for qbit in range(N_qubit):
        new_nonzero_values = []
        new_nonzero_locations = []

        gate_operations = {}
        sparsity = len(nonzero_locations)

        phases: np.ndarray = np.angle(nonzero_values)

        i = 0
        while i in range(sparsity):
            angle: float
            phase: float

            loc = nonzero_locations[i]

            # last step of the while loop
            if i + 1 == sparsity:
                new_nonzero_locations.append(loc // 2)
                if nonzero_locations[i] % 2 == 0:
                    # if the non_zero element is at the very end of the vector
                    angle = 0.0
                    phase = -phases[i]
                    new_nonzero_values.append(nonzero_values[i])
                else:
                    # if the non_zero element is second-last
                    angle = np.pi
                    phase = phases[i]
                    new_nonzero_values.append(abs(nonzero_values[i]))
            else:
                # divide the non_zero locations in pairs
                loc0 = nonzero_locations[i]
                loc1 = nonzero_locations[i + 1]

                # if the non_zero locations are consecutive, with the first one in an even position
                if (loc1 - loc0 == 1) and (loc0 % 2 == 0):
                    new_component = np.exp(1j * phases[i]) * np.sqrt(
                        abs(nonzero_values[i]) ** 2 + abs(nonzero_values[i + 1]) ** 2
                    )
                    new_nonzero_values.append(new_component)
                    new_nonzero_locations.append(loc0 // 2)

                    angle = (
                        2
                        * np.arccos(
                            np.clip(abs(nonzero_values[i] / new_component), -1, 1)
                        )
                        if abs(new_component) > _ZERO
                        else 0.0
                    )
                    phase = -phases[i] + phases[i + 1]
                    i += 1
                else:
                    # the non_zero location is on the right of the pair
                    if loc0 % 2 == 0:
                        angle = 0.0
                        phase = -phases[i]
                        new_nonzero_values.append(nonzero_values[i])
                        new_nonzero_locations.append(loc0 // 2)

                    else:
                        angle = np.pi
                        phase = phases[i]
                        new_nonzero_values.append(abs(nonzero_values[i]))
                        new_nonzero_locations.append(loc0 // 2)

            i += 1

            # add in the dictionary gate_operations if they are not zero
            if abs(angle) > _ZERO or abs(phase) > _ZERO:
                # number of control qubits for the current rotation gates
                num_controls = N_qubit - qbit - 1
                gate = (angle, phase)

                if num_controls == 0:
                    gate_operations = {"": gate}
                else:
                    controls = str(bin(loc // 2)[2:]).zfill(num_controls)
                    gate_operations[controls] = gate

        nonzero_values, nonzero_locations = (new_nonzero_values, new_nonzero_locations)

        if optimization:
            gate_operations = _optimize_dict(gate_operations)

        final_gates.append(gate_operations)

    final_gates.reverse()
    return final_gates


def _sanitize_sparse_state_vector(vec, *, copy=True):
    """given a list of complex numbers, build a normalized state vector stored as a scipy CSR matrix"""

    vec = sp.sparse.csr_matrix(vec)
    if copy:
        vec = vec.copy()

    vec /= sp.linalg.norm(vec.data)  # normalize
    vec.sort_indices()  # order non-zero locations

    return vec


def _number_of_qubits(vec: int) -> int:
    """number of qubits needed to represent the vector/vector size."""
    sz: int = vec if isinstance(vec, int) else len(vec)
    return int(np.ceil(np.log2(sz)))


def _optimize_dict(
    gate_operations,
):
    """
    Optimize the dictionary by merging some gates in one:
    if the two values are the same and they only differ in one control (one char of the key  is 0 and the other is 1) they can be merged
    >> {'11':[3.14,0] ; '10':[3.14,0]} becomes {'1e':[3.14,0]} where 'e' means no control (identity)

    >>> assert optimize_dict({"11": (3.14, 0), "10": (3.14, 0)}) == {"1e": (3.14, 0)}

    Args:
        gate_operations: collection of controlled gates to be applied
    Returns:
        optimized collection of controlled gates
    """
    while _run_one_merge_step(gate_operations):
        pass
    return gate_operations


def _run_one_merge_step(
    gate_operations,
):
    """
    Run a single merging step, modifying the input dictionary.

    Args:
        gate_operations: collection of controlled gates to be applied
    Returns:
        True if some merge happened
    """
    if len(gate_operations) <= 1:
        return False

    for k1, v1 in gate_operations.items():
        neighbours = _neighbour_dict(k1)

        for k2, position in neighbours.items():
            if k2 not in gate_operations:
                continue

            v2 = gate_operations[k2]

            # Consider only different items with same angle and phase
            if (abs(v1[0] - v2[0]) > _ZERO) or (abs(v1[1] - v2[1]) > _ZERO):
                continue

            # Replace the different char with 'e' and remove the old items
            gate_operations.pop(k1)
            gate_operations.pop(k2)
            gate_operations[k1[:position] + "e" + k1[position + 1 :]] = v1
            return True

    return False


def _neighbour_dict(controls):
    """
    Finds the neighbours of a string (ignoring e), i.e. the mergeble strings
    Returns a dictionary with as keys the neighbours and as value the position in which they differ

    >>> assert neighbour_dict("10") == {"00": 0, "11": 1}
    >>> assert neighbour_dict("1e") == {'0e': 0}

    Args:
        controls: string made of '0', '1', 'e'
    Returns:
        A dictionary {control-string: swapped-index}
    """
    neighbours = {}
    for i, c in enumerate(controls):
        if c == "e":
            continue

        c_opposite = "1" if c == "0" else "0"
        key = controls[:i] + c_opposite + controls[i + 1 :]
        neighbours[key] = i

    return neighbours
