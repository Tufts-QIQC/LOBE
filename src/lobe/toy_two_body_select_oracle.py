import cirq
import numpy as np

def add__toy_pairing_select_oracle(circuit, validation, control, index_register, system, operators):

    control_qubits = [validation] + index_register + system
    circuit.append(cirq.X.on(control).controlled_by(*control_qubits, control_values= [1, 0, 0, 0, 1, 1]))

    circuit.append(cirq.X.on(system[0]).controlled_by(control))
    circuit.append(cirq.X.on(system[1]).controlled_by(control))
    circuit.append(cirq.X.on(system[2]).controlled_by(control))
    circuit.append(cirq.X.on(system[3]).controlled_by(control))

    circuit.append(cirq.X.on(validation).controlled_by(control))
    circuit.append(cirq.X.on(control).controlled_by(*([validation] + index_register), control_values = [0, 0]))

    return circuit   