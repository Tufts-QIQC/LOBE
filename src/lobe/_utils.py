import numpy as np
from typing import List
import openparticle as op


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
