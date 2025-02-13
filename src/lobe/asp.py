import cirq
import numpy as np
from ._grover_rudolph import _grover_rudolph
from .multiplexed_rotations import get_decomposed_multiplexed_rotation_circuit
from .metrics import CircuitMetrics


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


def add_prepare_circuit(
    qubits, target_state, dagger=False, clean_ancillae=[], ctrls=([], [])
):
    """Add a quantum circuit that prepares the target state (arbitrary quantum state) when acting on the all-zero state.

    Implementation based on: https://arxiv.org/abs/quant-ph/0208112

    Args:
        qubits (List[cirq.LineQubit]): The qubits that the circuit acts upon
        target_state (np.ndarray): A numpy array describing the arbitrary quantum state
        dagger (bool): Flag to determine if daggered operation is desired
        clean_ancillae (List[cirq.LineQubit]): A list of qubits that are promised to start and end in the 0-state.
        ctrls (Tuple(List[cirq.LineQubit], List[int])): A set of qubits and integers that correspond to
            the control qubits and values.

    Returns:
        - List of cirq operations representing the gates to be applied in the circuit
        - CircuitMetrics object representing cost of block-encoding circuit
    """
    reordered_target_state = np.zeros(1 << len(qubits), dtype=complex)
    for i, coeff in enumerate(target_state):
        bitstring = format(i, f"0{2+len(qubits)}b")[2:]
        reordered_target_state[int(bitstring[::-1], 2)] = coeff

    gates = []
    metrics = CircuitMetrics()
    gate_list = _grover_rudolph(target_state)

    if dagger:
        gate_list = gate_list[::-1]

    multiplexing_angles = []
    for _ in range((len(qubits))):
        multiplexing_angles.append([])

    # index of list is qubits that gates act on; item in list is a dictionary with gate instructions
    for qubit_index, instructions in enumerate(gate_list):

        instructions = list(instructions.items())
        for controls, (ry_angle, rz_angle) in instructions:

            if not np.isclose(rz_angle, 0):
                raise RuntimeError(
                    "Our implementation assumes amplitudes are real-valued, therefore angle of rz gates should always be 0."
                )
            control_values = [int(ctrl) for ctrl in controls]
            multiplexing_angles[len(control_values)].append(ry_angle)

    if multiplexing_angles == [[]]:
        return [], CircuitMetrics()

    if dagger:
        multiplexing_angles[0][0] *= -1

    angle = multiplexing_angles[0][0]
    gates.append(
        cirq.ry(angle).on(qubits[0]).controlled_by(*ctrls[0], control_values=ctrls[1])
    )
    if len(ctrls[0]) > 0:
        metrics.rotation_angles.append(-angle / 2)
        metrics.rotation_angles.append(angle / 2)
    else:
        metrics.rotation_angles.append(angle)

    for qubit_index, angles in enumerate(multiplexing_angles[1:]):
        rotation_gates, rotation_metrics = get_decomposed_multiplexed_rotation_circuit(
            qubits[: qubit_index + 1],
            qubits[qubit_index + 1],
            angles,
            dagger=dagger,
            clean_ancillae=clean_ancillae,
            ctrls=ctrls,
        )
        gates += rotation_gates
        metrics += rotation_metrics

    if dagger:
        gates = gates[::-1]

    return gates, metrics
