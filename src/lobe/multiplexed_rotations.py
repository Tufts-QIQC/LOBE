import numpy as np
import cirq


def get_decomposed_multiplexed_rotation_circuit(
    register, angles, clean_ancillae=[], ctrls=([], []), numerics=None, dagger=False
):
    """Get the operations to add multiplexed rotations based on arXiv:0407010.

    Args:
        register (cirq.LineQubit): The qubit register on which the multiplexed rotations occur on.
            The qubit at index -1 is assumed to be the qubit that the rotations are applied upon.
        angles (np.array): A list of the rotation angles (alpha_i in arXiv:0407010)
        ctrls (Tuple(List[cirq.LineQubit], List[int])): A set of qubits and integers that correspond to
            the control qubits and values.
    Returns:
        - List[cirq.Moment]: A list of the circuit operations required for implementing multiplexed
            rotations
    """
    angles = np.concatenate([angles, np.zeros((1 << len(register) - 1) - len(angles))])
    processed_angles = _process_rotation_angles(angles)
    if dagger:
        processed_angles *= -1

    gates = _recursive_helper(
        register,
        processed_angles,
        0,
        len(register) - 1,
        clean_ancillae=clean_ancillae,
        ctrls=ctrls,
    )

    if (len(ctrls[0]) > 0) and (len(clean_ancillae) > 0):
        gates.append(
            cirq.X.on(clean_ancillae[0]).controlled_by(
                register[0], *ctrls[0], control_values=[1] + ctrls[1]
            )
        )
        gates.append(cirq.X.on(register[-1]).controlled_by(clean_ancillae[0]))
        gates.append(
            cirq.X.on(clean_ancillae[0]).controlled_by(
                register[0], *ctrls[0], control_values=[1] + ctrls[1]
            )
        )
    else:
        gates.append(
            cirq.X.on(register[-1]).controlled_by(
                register[0], *ctrls[0], control_values=[1] + ctrls[1]
            )
        )

    if len(ctrls[0]) > 0:
        gates.append(
            cirq.ry(-np.pi * sum(processed_angles))
            .on(register[-1])
            .controlled_by(*ctrls[0], control_values=[not val for val in ctrls[1]])
        )

    if numerics is not None:
        # Using decomposed ctrld-multiplexed rotations
        numerics["rotations"] += (len(processed_angles)) + 2
        numerics["angles"] += processed_angles.tolist()
        numerics["angles"].append(np.pi * sum(processed_angles) / 2)
        numerics["angles"].append(-np.pi * sum(processed_angles) / 2)
        numerics["left_elbows"] += len(processed_angles)
        numerics["right_elbows"] += len(processed_angles)
        numerics["ancillae_tracker"].append(numerics["ancillae_tracker"][-1] + 1)
        numerics["ancillae_tracker"].append(numerics["ancillae_tracker"][-1] - 1)
    return gates


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


def _recursive_helper(
    register, angles, rotation_index, level, clean_ancillae=[], ctrls=([], [])
):
    gates = []

    if level == 1:
        gates.append(cirq.ry(np.pi * angles[rotation_index]).on(register[-1]))

        if (len(ctrls[0]) > 0) and (len(clean_ancillae) > 0):
            gates.append(
                cirq.X.on(clean_ancillae[0]).controlled_by(
                    register[-2], *ctrls[0], control_values=[1] + ctrls[1]
                )
            )
            gates.append(cirq.X.on(register[-1]).controlled_by(clean_ancillae[0]))
            gates.append(
                cirq.X.on(clean_ancillae[0]).controlled_by(
                    register[-2], *ctrls[0], control_values=[1] + ctrls[1]
                )
            )
        else:
            gates.append(
                cirq.X.on(register[-1])
                .controlled_by(register[-2])
                .controlled_by(*ctrls[0], control_values=ctrls[1])
            )
        gates.append(cirq.ry(np.pi * angles[rotation_index + 1]).on(register[-1]))

    else:
        gates += _recursive_helper(
            register,
            angles,
            rotation_index,
            level - 1,
            clean_ancillae=clean_ancillae,
            ctrls=ctrls,
        )

        if (len(ctrls[0]) > 0) and (len(clean_ancillae) > 0):
            gates.append(
                cirq.X.on(clean_ancillae[0]).controlled_by(
                    register[-level - 1], *ctrls[0], control_values=[1] + ctrls[1]
                )
            )
            gates.append(cirq.X.on(register[-1]).controlled_by(clean_ancillae[0]))
            gates.append(
                cirq.X.on(clean_ancillae[0]).controlled_by(
                    register[-level - 1], *ctrls[0], control_values=[1] + ctrls[1]
                )
            )
        else:
            gates.append(
                cirq.X.on(register[-1])
                .controlled_by(register[-level - 1])
                .controlled_by(*ctrls[0], control_values=ctrls[1])
            )
        gates += _recursive_helper(
            register,
            angles,
            rotation_index + (1 << (level - 1)),
            level - 1,
            clean_ancillae=clean_ancillae,
            ctrls=ctrls,
        )

    return gates
