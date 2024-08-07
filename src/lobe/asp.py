import cirq
import numpy as np
from ._grover_rudolph import _grover_rudolph


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
