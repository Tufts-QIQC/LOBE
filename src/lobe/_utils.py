import cirq
import numpy as np
from openparticle import ParticleOperator, Fock, BosonOperator, FermionOperator
from typing import List


def pretty_print(
    wavefunction: np.ndarray,
    register_lengths: List[int],
    amplitude_cutoff=1e-12,
    decimal_places=3,
) -> str:
    """Get a human-readable description of the quantum state.

    Args:
        - wavefunction (np.ndarray): The amplitudes of the wavefunction
        - register_lengths (List[int]): The number of qubits in each separate register
        - amplitude_cutoff (float): The minimum mangitude for an amplitude to be considered nonzero
        - decimal_places (int): The number of decimal points to print

    Returns:
        - str: A string representing the wavefunction that can be printed
    """
    left_padding = 2 * (4 + decimal_places) + 1
    right_padding = sum(register_lengths) + len(register_lengths) + 1
    pretty_string = ""
    total_number_of_qubits = int(np.log2(len(wavefunction)))
    for state_index, amplitude in enumerate(wavefunction):
        if np.abs(amplitude) > amplitude_cutoff:

            state_bitstring = format(state_index, f"0{2+total_number_of_qubits}b")[2:]
            qubit_counter = 0
            ket_string = ""
            for register_length in register_lengths:

                state_of_register = state_bitstring[
                    qubit_counter : qubit_counter + register_length
                ]
                ket_string += "|" + state_of_register
                qubit_counter += register_length

            pretty_string += f"{str(amplitude.round(decimal_places)): <{left_padding}} {ket_string: >{right_padding}}>\n"

    return pretty_string


def get_basis_of_full_system(
    maximum_occupation_number,
    number_of_fermionic_modes=0,
    number_of_bosonic_modes=0,
):
    """Get the Fock basis of the system

    Args:
        - maximum_occupation_number (int): The maximum number of bosons allowed in each mode
        - number_of_fermionic_modes (int): The number of fermionic modes
        - number_of_bosonic_modes (int): The number of bosonic modes

    Returns:
        - List[Fock]: A list of Fock states
    """
    number_of_occupation_qubits = max(
        int(np.ceil(np.log2(maximum_occupation_number))), 1
    )

    total_number_of_qubits = number_of_fermionic_modes + (
        number_of_bosonic_modes * number_of_occupation_qubits
    )

    basis = []
    for basis_state in range(1 << total_number_of_qubits):
        qubit_values = [
            int(i) for i in format(basis_state, f"#0{2+total_number_of_qubits}b")[2:]
        ]

        index = 0
        fermionic_fock_state = []
        fermionic_fock_state = [
            i for i in range(number_of_fermionic_modes) if qubit_values[i] == 1
        ]
        index += number_of_fermionic_modes

        bosonic_fock_state = []
        for mode in range(number_of_bosonic_modes):
            occupation = ""
            for val in qubit_values[index : index + number_of_occupation_qubits]:
                occupation += str(val)
            bosonic_fock_state.append((mode, int(occupation, 2)))
            index += number_of_occupation_qubits

        basis.append(Fock(fermionic_fock_state, [], bosonic_fock_state))
    return basis


def get_active_bosonic_modes(operator):
    """Get a list of the bosonic modes being acted on.

    Args:
        operator (Optional[ParticleOperator, List[ParticleOperator]]): The operator/term in question

    Returns:
        List[int]: A list of active bosonic modes
    """
    active_modes = []
    for term in operator.to_list():
        for op in term.split():
            if isinstance(op, BosonOperator):
                if op.mode not in active_modes:
                    active_modes.append(op.mode)
    return active_modes


def get_active_fermionic_modes(operator):
    """Get a list of the fermionic modes being acted on.

    Args:
        operator (Optional[ParticleOperator, List[ParticleOperator]]): The operator/term in question

    Returns:
        List[int]: A list of active fermionic modes
    """
    active_modes = []
    for term in operator.to_list():
        for op in term.split():
            if isinstance(op, FermionOperator):
                if op.mode not in active_modes:
                    active_modes.append(op.mode)
    return active_modes


def get_number_of_active_bosonic_modes(terms):
    """Get a list of the number of bosonic modes being acted on in each term.

    Args:
        terms (List[ParticleOperator]): The terms comprising the original Hamiltonian
            given as a linear combination of ladder operators.

    Returns:
        List[int]: A list of the number of bosonic operators in each term
    """
    numbers_of_active_bosonic_modes = []
    for term in terms:
        active_modes = get_active_bosonic_modes(term)
        numbers_of_active_bosonic_modes.append(len(active_modes))

    return numbers_of_active_bosonic_modes


def _get_parsed_dictionary(operator, number_of_modes=None):
    """Helper function to parse active modes and exponents"""
    if number_of_modes is None:
        number_of_modes = operator.max_mode + 1
    parsed_operator_array = {
        "fermion": {
            "operator_types": [None] * number_of_modes,
        },
        "boson": {
            "creation": np.zeros(number_of_modes, dtype=int),
            "annihilation": np.zeros(number_of_modes, dtype=int),
        },
    }
    for ladder in operator.split():
        if isinstance(ladder, FermionOperator):
            if (
                parsed_operator_array[ladder.particle_type]["operator_types"][
                    ladder.mode
                ]
                is None
            ):
                if ladder.creation:
                    parsed_operator_array[ladder.particle_type]["operator_types"][
                        ladder.mode
                    ] = 1
                else:
                    parsed_operator_array[ladder.particle_type]["operator_types"][
                        ladder.mode
                    ] = 0
            else:
                if (
                    parsed_operator_array[ladder.particle_type]["operator_types"][
                        ladder.mode
                    ]
                    == 1
                ) and (not ladder.creation):
                    parsed_operator_array[ladder.particle_type]["operator_types"][
                        ladder.mode
                    ] = 2
                elif (
                    parsed_operator_array[ladder.particle_type]["operator_types"][
                        ladder.mode
                    ]
                    == 0
                ) and (ladder.creation):
                    parsed_operator_array[ladder.particle_type]["operator_types"][
                        ladder.mode
                    ] = 3
                else:
                    raise RuntimeError(
                        f"Operator: {operator} not mode ordered. Problem mode: {ladder.mode}"
                    )
        elif isinstance(ladder, BosonOperator):
            if ladder.creation:
                parsed_operator_array[ladder.particle_type]["creation"][
                    ladder.mode
                ] += 1
            else:
                parsed_operator_array[ladder.particle_type]["annihilation"][
                    ladder.mode
                ] += 1
        else:
            raise RuntimeError(
                "Unknown operator: {} with particle type: {}".format(
                    type(ladder), ladder.particle_type
                )
            )
    return parsed_operator_array


def get_bosonic_exponents(operator, number_of_modes=None):
    """Get exponents of bosonic ladder operators acting on unique modes within one term

    NOTE:
        Example: a_0^\dagger a_0^\dagger a_0 a_1 -> ([0, 1], [(2, 1), (0, 1)])

    Args:
        - operator (ParticleOperator): A single term to parse out the bosonic operators
        - number_of_modes (Optional[int]): The total number of modes in the operator

    Returns:
        - List[int]: A list of the active bosonic modes
        - List[Tuple(int, int)]: A list of tuples corresponding to the exponents on each active mode. Each tuple
            contains the exponent on the creation operator, followed by the exponent on the annihilation operator
    """
    active_bosonic_modes = get_active_bosonic_modes(operator)
    parsed_operator_array = _get_parsed_dictionary(
        operator, number_of_modes=number_of_modes
    )
    exponents_list = []
    for active_mode in active_bosonic_modes:
        exponents_list.append(
            (
                parsed_operator_array["boson"]["creation"][active_mode],
                parsed_operator_array["boson"]["annihilation"][active_mode],
            )
        )
    return active_bosonic_modes, exponents_list


def get_fermionic_operator_types(operator, number_of_modes=None):
    """Get operator types of fermionic ladder operators acting on unique modes within one term

    NOTE:
        Example: a_0^\dagger a_0^\dagger a_0 a_1 -> ([0, 1], [(2, 1), (0, 1)])

    Args:
        - operator (ParticleOperator): A single term to parse out the fermionic operators
        - number_of_modes (Optional[int]): The total number of modes in the operator

    Returns:
        - List[int]: A list of the active fermionic modes
        - List[int]: A list of tuples corresponding to the operator types on each active mode. Each tuple
            contains the exponent on the creation operator, followed by the exponent on the annihilation operator
    """
    active_fermionic_modes = get_active_fermionic_modes(operator)
    parsed_operator_array = _get_parsed_dictionary(
        operator, number_of_modes=number_of_modes
    )
    operator_types = []
    for active_mode in active_fermionic_modes:
        operator_types.append(
            parsed_operator_array["fermion"]["operator_types"][active_mode]
        )
    return active_fermionic_modes, operator_types


def _apply_negative_identity(target, ctrls=([], [])):
    """Add a controlled -I to the circuit

    Args:
        target (cirq.LineQubit): An arbitrary qubit on which to apply the -I
         ctrls (Tuple(List[cirq.LineQubit], List[int])): A set of qubits and integers that correspond to
             the control qubits and values.

    Returns:
        - List of cirq operations representing the gates to be applied in the circuit
    """
    gates = []

    gates.append(cirq.X.on(target).controlled_by(*ctrls[0], control_values=ctrls[1]))
    gates.append(cirq.Y.on(target).controlled_by(*ctrls[0], control_values=ctrls[1]))
    gates.append(cirq.Z.on(target).controlled_by(*ctrls[0], control_values=ctrls[1]))
    gates.append(cirq.X.on(target).controlled_by(*ctrls[0], control_values=ctrls[1]))
    gates.append(cirq.Y.on(target).controlled_by(*ctrls[0], control_values=ctrls[1]))
    gates.append(cirq.Z.on(target).controlled_by(*ctrls[0], control_values=ctrls[1]))

    return gates


def translate_antifermions_to_fermions(operator):
    """Translate all antifermionic modes to fermionic modes with higher index

    Args:
        - operator (ParticleOperator): The operation which potentially contains antifermions

    Returns:
        - ParticleOperator: The operator where antifermionic modes are replaced with distinct fermionic modes
    """
    translated_operator = None
    for term in operator:
        translated_term = None
        for op in term.split():
            new_op = op
            if list(op.op_dict.keys())[0][0][0] == 1:
                expected_tuple = (
                    0,
                    op.mode + operator.max_fermionic_mode + 1,
                    list(op.op_dict.keys())[0][0][2],
                )
                new_op = ParticleOperator({(expected_tuple,): op.coeff})
            if translated_term is None:
                translated_term = new_op
            else:
                translated_term *= new_op
        if translated_operator is None:
            translated_operator = translated_term
        else:
            translated_operator += translated_term
    return translated_operator


def predict_number_of_block_encoding_ancillae(operator):
    assert len(operator.to_list()) <= 2
    number_of_block_encoding_ancillae = 0

    number_of_block_encoding_ancillae = len(get_active_bosonic_modes(operator))
    has_fermionic_modes = len(get_active_fermionic_modes(operator)) > 0
    if (len(operator.to_list()) == 2) and (not has_fermionic_modes):
        # require additional index qubit between terms
        number_of_block_encoding_ancillae += 1
    elif has_fermionic_modes:
        number_of_block_encoding_ancillae += 1

    return number_of_block_encoding_ancillae
