import cirq
import numpy as np
from grover_rudolph.Algorithm.state_preparation import grover_rudolph


def add_coefficient_oracle(
    circuit,
    rotation_qubit,
    index_qubits,
    normalized_term_coefficients,
    number_of_operators,
):
    for index in range(number_of_operators):
        control_values = [
            int(i) for i in format(index, f"#0{2+len(index_qubits)}b")[2:]
        ][::-1]
        circuit.append(
            cirq.ry(2 * np.arccos(normalized_term_coefficients[index]))
            .on(rotation_qubit)
            .controlled_by(*index_qubits, control_values=control_values)
        )
    return circuit


def add_prepare_circuit(circuit, qubits, target_state, dagger=False):
    """Add a quantum circuit that prepares the target state (arbitrary quantum state) when acting on the all-zero state.

    Implementation based on: https://arxiv.org/abs/quant-ph/0208112

    Args:
        circuit (cirq.Circuit): The quantum circuit to add instructions to
        qubits (List[cirq.LineQubit]): The qubits that the circuit acts upon
        target_state (np.ndarray): A numpy array describing the arbitrary quantum state

    Returns:
        cirq.Circuit: The quantum circuit including the prepare oracle
    """

    gate_list = grover_rudolph(target_state)

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
                circuit.append(
                    cirq.ry(ry_angle)
                    .on(qubits[qubit_index])
                    .controlled_by(*qubits[:qubit_index], control_values=control_values)
                )

            # Controlled-Phase
            circuit.append(
                phase_gate.on(qubits[qubit_index]).controlled_by(
                    *qubits[:qubit_index], control_values=control_values
                )
            )

            if dagger:
                # Controlled-Ry
                circuit.append(
                    cirq.ry(ry_angle)
                    .on(qubits[qubit_index])
                    .controlled_by(*qubits[:qubit_index], control_values=control_values)
                )

    return circuit
