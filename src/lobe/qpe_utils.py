import cirq
import numpy as np

# Just some faster code to sample the measurements of the phase register in the infinite sample limit
def simulate_circuit_and_compute_probabilities(circuit, initial_state, bop):
    state = cirq.Simulator().simulate(circuit, initial_state=initial_state).final_state_vector
    traced_phase_reg_probabilities = np.zeros(1<<bop)
    for index in range(1 << len(circuit.all_qubits())):
        amp = state[index]
        prob = np.abs(amp)**2
        full_basis_state = format(index, "0{}b".format(len(circuit.all_qubits())))
        state_of_phase_reg = full_basis_state[:bop]
        traced_phase_reg_probabilities[int(state_of_phase_reg, 2)] += prob
    traced_phase_reg_probabilities = np.concatenate([traced_phase_reg_probabilities[-1<<(bop-1):], traced_phase_reg_probabilities[:1<<(bop-1)]])
    return traced_phase_reg_probabilities

# Relationship between measurement outcomes and eigenphase
def list_possible_eigenphases(bop):
    eigenphases = []
    for index in range(1 << bop):
        eigenphase = (index / (1 << (bop-1)))
        if eigenphase >= 1:
            eigenphase -= 2
        eigenphases.append(eigenphase)
    return sorted(eigenphases)

def expected_value_of_eigenphase(eigenphases, phase_reg_probabilities):
    exp_val = 0
    for eigenphase, probability in zip(eigenphases, phase_reg_probabilities):
        exp_val += eigenphase*probability
    return exp_val

def decode_eigenphase(eigenphase, alpha):
    return -alpha * np.cos(eigenphase*np.pi)