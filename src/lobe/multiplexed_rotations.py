import numpy as np
import cirq
from .metrics import CircuitMetrics

CLIFFORD_ROTATION_ANGLES = [i * np.pi / 2 for i in range(9)]


def get_decomposed_multiplexed_rotation_circuit(
    indexing_register,
    rotation_qubit,
    angles,
    dagger=False,
    clean_ancillae=[],
    ctrls=([], []),
):
    """Get the operations to add multiplexed rotations based on arXiv:0407010.

    Args:
        indexing_register (List[cirq.LineQubit]): The qubit register on which the multiplexed rotations are indexed over.
        rotation_qubit (cirq.LineQubit): The qubit on which the rotations are applied.
        angles (np.array): A list of the rotation angles (alpha_i in arXiv:0407010)
        dagger (bool): Flag to indicate if the circuit should be of the daggered form.
        clean_ancillae (List[cirq.LineQubit]): A list of qubits that are promised to start and end in the 0-state.
        ctrls (Tuple(List[cirq.LineQubit], List[int])): A set of qubits and integers that correspond to
            the control qubits and values.
    Returns:
        - List[cirq.Moment]: A list of the circuit operations required for implementing multiplexed
            rotations
        - CircuitMetrics object representing cost of block-encoding circuit
    """
    assert len(ctrls[0]) <= 1
    gates = []
    rotation_gadget_metrics = CircuitMetrics()
    angles = np.concatenate(
        [angles, np.zeros((1 << len(indexing_register)) - len(angles))]
    )
    processed_angles = _process_rotation_angles(angles)
    if dagger:
        processed_angles *= -1

    ancilla = []
    if len(ctrls[0]) > 0:
        rotation_gadget_metrics.number_of_elbows += len(ctrls[0]) - 1
        rotation_gadget_metrics.add_to_clean_ancillae_usage(len(ctrls[0]) - 1)

        for index_qubit in indexing_register:
            rotation_gadget_metrics.add_to_clean_ancillae_usage(1)
            rotation_gadget_metrics.number_of_elbows += 1
            gates.append(
                cirq.Moment(
                    cirq.X.on(clean_ancillae[len(ancilla)]).controlled_by(
                        *ctrls[0], index_qubit, control_values=ctrls[1] + [1]
                    )
                )
            )
            ancilla.append(clean_ancillae[len(ancilla)])

    for angle in processed_angles:
        if not np.any(
            [
                np.isclose((angle) % (4 * np.pi), clifford_angle)
                for clifford_angle in CLIFFORD_ROTATION_ANGLES
            ]
        ):
            # Count only nonClifford rotations
            rotation_gadget_metrics.number_of_rotations += 1

    _gates, recursion_metrics = _recursive_helper(
        indexing_register,
        rotation_qubit,
        processed_angles,
        0,
        len(indexing_register),
        ancilla=ancilla,
        clean_ancillae=clean_ancillae[len(ancilla) :],
        ctrls=ctrls,
    )
    gates += _gates
    rotation_gadget_metrics += recursion_metrics

    if len(ctrls[0]) > 0:
        gates.append(cirq.Moment(cirq.X.on(rotation_qubit).controlled_by(ancilla[0])))
    else:
        gates.append(
            cirq.Moment(
                cirq.X.on(rotation_qubit).controlled_by(
                    indexing_register[0], *ctrls[0], control_values=[1] + ctrls[1]
                )
            )
        )

    for index_qubit, anc in zip(indexing_register[::-1], ancilla[::-1]):
        rotation_gadget_metrics.add_to_clean_ancillae_usage(-1)
        gates.append(
            cirq.Moment(
                cirq.X.on(anc).controlled_by(
                    *ctrls[0], index_qubit, control_values=ctrls[1] + [1]
                )
            )
        )

    if len(ctrls[0]) > 0:
        angle = -np.pi * sum(processed_angles)
        if not np.any(
            [
                np.isclose((angle / 2) % (4 * np.pi), clifford_angle)
                for clifford_angle in CLIFFORD_ROTATION_ANGLES
            ]
        ):
            # Controlled rotations are implemented with two rotations of angle/2.
            # This checks if those rotations will be nonClifford
            rotation_gadget_metrics.number_of_rotations += 2
        gates.append(
            cirq.Moment(
                cirq.ry(angle)
                .on(rotation_qubit)
                .controlled_by(*ctrls[0], control_values=[not val for val in ctrls[1]])
            )
        )

    if len(ctrls[0]) > 0:
        rotation_gadget_metrics.add_to_clean_ancillae_usage(-(len(ctrls[0]) - 1))

    return gates, rotation_gadget_metrics


def _recursive_helper(
    indexing_register,
    rotation_qubit,
    angles,
    rotation_index,
    level,
    ancilla=[],
    clean_ancillae=[],
    ctrls=([], []),
):
    if ctrls != ([], []):
        assert len(ancilla) == len(indexing_register)
    recursion_metrics = CircuitMetrics()
    gates = []

    if level == 1:
        gates.append(
            cirq.Moment(cirq.ry(np.pi * angles[rotation_index]).on(rotation_qubit))
        )

        if len(ctrls[0]) > 0:
            gates.append(
                cirq.Moment(cirq.X.on(rotation_qubit).controlled_by(ancilla[-1]))
            )
        else:
            gates.append(
                cirq.Moment(
                    cirq.X.on(rotation_qubit)
                    .controlled_by(indexing_register[-1])
                    .controlled_by(*ctrls[0], control_values=ctrls[1])
                )
            )
        gates.append(
            cirq.Moment(cirq.ry(np.pi * angles[rotation_index + 1]).on(rotation_qubit))
        )

    else:
        additional_gates, additonal_metrics = _recursive_helper(
            indexing_register,
            rotation_qubit,
            angles,
            rotation_index,
            level - 1,
            ancilla=ancilla,
            clean_ancillae=clean_ancillae,
            ctrls=ctrls,
        )
        gates += additional_gates
        recursion_metrics += additonal_metrics

        if len(ctrls[0]) > 0:
            gates.append(
                cirq.Moment(cirq.X.on(rotation_qubit).controlled_by(ancilla[-level]))
            )
        else:
            gates.append(
                cirq.Moment(
                    cirq.X.on(rotation_qubit)
                    .controlled_by(indexing_register[-level])
                    .controlled_by(*ctrls[0], control_values=ctrls[1])
                )
            )
        additional_gates, additonal_metrics = _recursive_helper(
            indexing_register,
            rotation_qubit,
            angles,
            rotation_index + (1 << (level - 1)),
            level - 1,
            ancilla=ancilla,
            clean_ancillae=clean_ancillae,
            ctrls=ctrls,
        )
        gates += additional_gates
        recursion_metrics += additonal_metrics

    return gates, recursion_metrics


def _binary_to_gray(n):
    # Code from: https://www.sanfoundry.com/python-program-convert-binary-gray-code/
    """Convert Binary to Gray codeword and return it."""
    number_of_bits = len(n)
    n = int(n, 2)  # convert to int
    n ^= n >> 1

    # bin(n) returns n's binary representation with a '0b' prefixed
    # the slice operation is to remove the prefix
    gray_string = bin(n)[2:]
    gray_string = "0" * (number_of_bits - len(gray_string)) + gray_string
    return gray_string


def _binary_dot_product_mod_2(b, g):
    counter = 0
    for bi, gi in zip(b, g):
        if bi == "1" and gi == "1":
            counter += 1

    return counter % 2


def _generate_m_matrix(dimension):
    num_bits = int(np.log2(dimension))
    matrix = np.zeros((dimension, dimension))
    for row_index in range(dimension):
        g_string = _binary_to_gray(format(row_index, f"0{2+num_bits}b")[2:])
        for col_index in range(dimension):
            b_string = format(col_index, f"0{2+num_bits}b")[2:]
            matrix[row_index][col_index] = (1 / dimension) * (-1) ** (
                _binary_dot_product_mod_2(b_string, g_string)
            )
    return matrix


def _process_rotation_angles(angles):
    transformation = _generate_m_matrix(1 << int(np.ceil(np.log2(len(angles)))))
    return transformation @ angles
