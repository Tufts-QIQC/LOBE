import cirq


def add_naive_usp(circuit, index_register):
    for qubit in index_register:
        circuit.append(cirq.H.on(qubit))
    return circuit
