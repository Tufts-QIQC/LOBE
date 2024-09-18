import pytest
import numpy as np
import cirq
from src.lobe.asp import (
    add_prepare_circuit,
    get_target_state,
    get_multiplexed_grover_rudolph_circuit,
    get_grover_rudolph_instructions,
)
from src.lobe._grover_rudolph import _ZERO


def _wrapper_func(qubits, target_state, dagger=False):
    instructions = get_grover_rudolph_instructions(target_state)
    clean_ancillae = qubits
    qubits = [cirq.LineQubit(i + len(qubits)) for i in range(len(qubits))]
    return get_multiplexed_grover_rudolph_circuit(
        instructions, qubits, clean_ancillae, dagger=dagger
    )


@pytest.mark.parametrize("number_of_qubits", range(1, 6))
@pytest.mark.parametrize("number_of_terms", np.random.random_integers(1 << 10, size=20))
@pytest.mark.parametrize("asp_func", [_wrapper_func, add_prepare_circuit])
def test_asp(number_of_qubits, number_of_terms, asp_func):
    number_of_terms = number_of_terms % (1 << number_of_qubits)
    if number_of_terms == 0:
        return
    number_of_qubits = max(int(np.ceil(np.log2(number_of_terms))), 1)

    coefficients = np.random.uniform(0, 100, size=number_of_terms)
    target_state = get_target_state(coefficients)

    circuit = cirq.Circuit()
    qubits = [cirq.LineQubit(i) for i in range(number_of_qubits)]

    circuit.append(cirq.I.on_each(*qubits))
    circuit += asp_func(qubits, target_state)

    simulator = cirq.Simulator()
    wavefunction = simulator.simulate(circuit).final_state_vector
    for index, coeff in enumerate(target_state):
        assert np.isclose(wavefunction[index], coeff, atol=100 * _ZERO)

    circuit += asp_func(qubits, target_state, dagger=True)
    wavefunction = simulator.simulate(circuit).final_state_vector
    one_state = np.zeros(1 << number_of_qubits)
    one_state[0] = 1
    for index, coeff in enumerate(one_state):
        assert np.isclose(wavefunction[index], coeff, atol=100 * _ZERO)
