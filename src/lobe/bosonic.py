from .metrics import CircuitMetrics
import cirq
import numpy as np
from src.lobe.addition import add_classical_value_gate_efficient
from src.lobe.multiplexed_rotations import get_decomposed_multiplexed_rotation_circuit


def _add_multi_bosonic_rotations(
    rotation_qubit,
    bosonic_mode_register,
    creation_exponent=0,
    annihilation_exponent=0,
    clean_ancillae=[],
    ctrls=([], []),
    numerics=None,
):
    """Add rotations to pickup bosonic coefficients corresponding to a series of ladder operators (assumed
        to be normal ordered) acting on one bosonic mode within a term.

    Args:
        rotation_qubit (cirq.LineQubit): The qubit that is rotated to pickup the amplitude corresponding
            to the coefficients that appear when a bosonic op hits a quantum state
        bosonic_mode_register (List[cirq.LineQubit]): The qubits that store the occupation of the bosonic
            mode being acted upon.
        creation_exponent (int): The number of subsequent creation operators in the term
        annihilation_exponent (int): The number of subsequent annihilation operators in the term
        ctrls (Tuple(List[cirq.LineQubit], List[int])): A set of qubits and integers that correspond to
            the control qubits and values.

    Returns:
        - The gates to perform the unitary operation
    """
    gates = []

    maximum_occupation_number = (1 << len(bosonic_mode_register)) - 1

    # Flip the rotation qubit outside the encoded subspace
    gates.append(
        cirq.Moment(
            cirq.X.on(rotation_qubit).controlled_by(*ctrls[0], control_values=ctrls[1])
        )
    )

    # Multiplexing over computational basis states of mode register that will not be zeroed-out
    angles = []
    for particle_number in range(
        0,
        maximum_occupation_number + 1,
    ):
        if (particle_number - creation_exponent) < 0:
            angles.append(0)
        elif (
            particle_number - creation_exponent + annihilation_exponent
        ) > maximum_occupation_number:
            angles.append(0)
        else:
            # Classically compute coefficient that should appear
            intended_coefficient = 1
            for power in range(creation_exponent):
                intended_coefficient *= np.sqrt(
                    (particle_number - power) / (maximum_occupation_number + 1)
                )
            for power in range(annihilation_exponent):
                intended_coefficient *= np.sqrt(
                    (particle_number - creation_exponent + power + 1)
                    / (maximum_occupation_number + 1)
                )
            angles.append(2 * np.arcsin(-1 * intended_coefficient) / np.pi)

    gates += get_decomposed_multiplexed_rotation_circuit(
        bosonic_mode_register + [rotation_qubit],
        angles,
        clean_ancillae=clean_ancillae,
        ctrls=ctrls,
        numerics=numerics,
    )
    return gates


def bosonic_mode_block_encoding(
    system,
    block_encoding_ancilla,
    active_index,
    exponents,
    clean_ancillae=[],
    ctrls=([], []),
):
    """Obtain operations for a block-encoding circuit for a product of bosonic ladder operators acting on the same mode

    NOTE: Assumes operator is written in the form: $(a_i^\dagger)^R (a_i)^S)$

    Args:
        system (lobe.system.System): The system object holding the system registers
        block_encoding_ancilla (cirq.LineQubit): The single block-encoding ancilla qubit
        active_index (int): An integer representing the bosonic mode on which the operator acts
        exponents (Tuple(int, int)): A Tuple of ints corresponds to the exponents of the operators: (S, R).
        clean_ancillae (List[cirq.LineQubit]): A list of qubits that are promised to start and end in the 0-state.
        ctrls (Tuple(List[cirq.LineQubit], List[int])): A set of qubits and integers that correspond to
            the control qubits and values.

    Returns:
        - List of cirq operations representing the gates to be applied in the circuit
        - CircuitMetrics object representing cost of block-encoding circuit
    """

    gates = []
    block_encoding_metrics = CircuitMetrics()

    # gates.append(cirq.H.on(block_encoding_ancilla))
    R, S = exponents[0], exponents[1]

    adder_controls = (ctrls[0] + [block_encoding_ancilla], ctrls[1] + [0])
    gates += add_classical_value_gate_efficient(
        system.bosonic_system[active_index],
        R - S,
        clean_ancillae=clean_ancillae,
        ctrls=adder_controls,
    )

    gates += _add_multi_bosonic_rotations(
        block_encoding_ancilla,
        system.bosonic_system[active_index],
        R,
        S,
        clean_ancillae=clean_ancillae,
        ctrls=ctrls,
    )

    # gates.append(cirq.H.on(block_encoding_ancilla))

    return gates, block_encoding_metrics


def bosonic_mode_plus_hc_block_encoding(
    system,
    block_encoding_ancilla,
    active_index,
    exponents,
    clean_ancillae=[],
    ctrls=([], []),
):
    """Obtain operations for a block-encoding circuit for a product of bosonic ladder ops plus hermitian conjugate

    NOTE: Input arguements should correspond to only one term of the form: $(a_i^\dagger)^R (a_i)^S)$.
        The form of the hermitian conjugate will be inferred.

    Args:
        system (lobe.system.System): The system object holding the system registers
        block_encoding_ancilla (cirq.LineQubit): The single block-encoding ancilla qubit
        active_index (int): An integer representing the bosonic mode on which the operator acts
        exponents (Tuple(int, int)): A Tuple of ints corresponds to the exponents of the operators: (S, R).
        clean_ancillae (List[cirq.LineQubit]): A list of qubits that are promised to start and end in the 0-state.
        ctrls (Tuple(List[cirq.LineQubit], List[int])): A set of qubits and integers that correspond to
            the control qubits and values.

    Returns:
        - List of cirq operations representing the gates to be applied in the circuit
        - CircuitMetrics object representing cost of block-encoding circuit
    """
    pass
