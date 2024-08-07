import pytest
import numpy as np
import cirq
from src.lobe.asp import add_prepare_circuit, get_target_state, _ZERO


@pytest.mark.parametrize("trial", range(100))
def test_asp(trial):
    number_of_qubits = np.random.random_integers(1, 5)
    number_of_terms = np.random.random_integers(1 << 10)
    number_of_terms = number_of_terms % (1 << number_of_qubits)
    if number_of_terms == 0:
        return
    number_of_qubits = max(int(np.ceil(np.log2(number_of_terms))), 1)

    coefficients = np.random.uniform(0, 100, size=number_of_terms)
    target_state = get_target_state(coefficients)

    circuit = cirq.Circuit()
    qubits = [cirq.LineQubit(i) for i in range(number_of_qubits)]

    circuit.append(cirq.I.on_each(*qubits))
    circuit += add_prepare_circuit(qubits, target_state)

    simulator = cirq.Simulator()
    wavefunction = simulator.simulate(circuit).final_state_vector
    assert np.allclose(wavefunction, target_state)

    circuit += add_prepare_circuit(qubits, target_state, dagger=True)
    wavefunction = simulator.simulate(circuit).final_state_vector
    one_state = np.zeros(1 << number_of_qubits)
    one_state[0] = 1
    assert np.allclose(wavefunction, one_state, atol=100 * _ZERO)
