def get_index_of_reversed_bitstring(integer: int, number_of_qubits: int) -> int:
    return int(format(integer, f"0{2+number_of_qubits}b")[2:][::-1], 2)
