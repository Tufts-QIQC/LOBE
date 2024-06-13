import cirq
import numpy as np


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
        ]
        circuit.append(
            cirq.ry(2 * np.arccos(normalized_term_coefficients[index]))
            .on(rotation_qubit)
            .controlled_by(*index_qubits, control_values=control_values)
        )
    return circuit
