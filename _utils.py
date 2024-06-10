import numpy as np
from typing import List


def get_index_of_reversed_bitstring(integer: int, number_of_qubits: int) -> int:
    return int(format(integer, f"0{2+number_of_qubits}b")[2:][::-1], 2)


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
                ][::-1]
                ket_string += "|" + state_of_register
                qubit_counter += register_length

            pretty_string += f"{str(amplitude.round(decimal_places)): <{left_padding}} {ket_string: >{right_padding}}>\n"

    return pretty_string
