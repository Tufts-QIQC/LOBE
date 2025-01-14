import numpy as np
from typing import List
import openparticle as op
from .rescale import get_active_bosonic_modes


def pretty_print(
    wavefunction: np.ndarray,
    register_lengths: List[int],
    amplitude_cutoff=1e-12,
    decimal_places=3,
) -> str:

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


def give_me_state(nn, Lambda, Omega):
    # Credit to Kamil Serafin
    n = nn
    result = list()
    for i in range(Lambda):
        r = n % (Omega + 1)
        n = n // (Omega + 1)
        result += [r]

    return list(enumerate(result))


def generate_bosonic_pairing_hamiltonian_matrix(mode_cutoff, occupancy_cutoff):
    """
    Generating bosonic pairing Hamiltonians to test LOBE

    Mode cutoff \Lambda

    H = a_p^\dagger a_q

    Set of states: \{|;;(0, \omega)(0, \omega') \rangle \} for omega, omega' \in Omega

    By choosing some occupancy cutoff, \Omega, this (with \Lambda) defines the size of the Hamiltonian matrix

    Matrix size goes as (Omega + 1) ** Lambda

    """

    H = op.ParticleOperatorSum([])
    for lambda_p in range(mode_cutoff):
        for lambda_q in range(mode_cutoff):
            op_string = "a" + str(lambda_p) + "^ " + "a" + str(lambda_q)
            H += op.ParticleOperator(op_string)
    basis = []

    for i in range((occupancy_cutoff + 1) ** mode_cutoff):
        state = op.Fock([], [], give_me_state(i, mode_cutoff, occupancy_cutoff))
        basis.append(state)

    matrix = op.utils.generate_matrix_from_basis(H, basis)

    return matrix


def get_basis_of_full_system(
    number_of_modes,
    maximum_occupation_number,
    has_fermions=False,
    has_antifermions=False,
    has_bosons=False,
):
    number_of_occupation_qubits = max(
        int(np.ceil(np.log2(maximum_occupation_number))), 1
    )

    total_number_of_qubits = 0
    if has_fermions:
        total_number_of_qubits += number_of_modes
    if has_antifermions:
        total_number_of_qubits += number_of_modes
    if has_bosons:
        total_number_of_qubits += number_of_modes * number_of_occupation_qubits

    basis = []
    for basis_state in range(1 << total_number_of_qubits):
        qubit_values = [
            int(i) for i in format(basis_state, f"#0{2+total_number_of_qubits}b")[2:]
        ]

        index = 0
        fermionic_fock_state = []
        if has_fermions:
            fermionic_fock_state = [
                i for i in range(number_of_modes) if qubit_values[i] == 1
            ]
            index += number_of_modes

        antifermionic_fock_state = []
        if has_antifermions:
            antifermionic_fock_state = [
                i for i in range(number_of_modes) if qubit_values[index + i] == 1
            ]
            index += number_of_modes

        bosonic_fock_state = []
        if has_bosons:
            for mode in range(number_of_modes):
                occupation = ""
                for val in qubit_values[index : index + number_of_occupation_qubits]:
                    occupation += str(val)
                bosonic_fock_state.append((mode, int(occupation, 2)))
                index += number_of_occupation_qubits

        basis.append(
            op.Fock(fermionic_fock_state, antifermionic_fock_state, bosonic_fock_state)
        )
    return basis


def get_parsed_dictionary(operator, number_of_modes=None):
    if number_of_modes is None:
        number_of_modes = operator.max_mode() + 1
    parsed_operator_array = {
        "fermion": {
            "creation": np.zeros(number_of_modes, dtype=int),
            "annihilation": np.zeros(number_of_modes, dtype=int),
        },
        "antifermion": {
            "creation": np.zeros(number_of_modes, dtype=int),
            "annihilation": np.zeros(number_of_modes, dtype=int),
        },
        "boson": {
            "creation": np.zeros(number_of_modes, dtype=int),
            "annihilation": np.zeros(number_of_modes, dtype=int),
        },
    }
    for ladder in operator.split():
        if ladder.creation:
            parsed_operator_array[ladder.particle_type]["creation"][ladder.mode] += 1
        else:
            parsed_operator_array[ladder.particle_type]["annihilation"][
                ladder.mode
            ] += 1
    return parsed_operator_array


def get_bosonic_exponents(operator, number_of_modes=None):
    active_bosonic_modes = get_active_bosonic_modes(operator)
    parsed_operator_array = get_parsed_dictionary(
        operator, number_of_modes=number_of_modes
    )
    exponents_list = []
    for active_mode in active_bosonic_modes:
        exponents_list.append(
            (
                parsed_operator_array["boson"]["annihilation"][active_mode],
                parsed_operator_array["boson"]["creation"][active_mode],
            )
        )
    return active_bosonic_modes, exponents_list
