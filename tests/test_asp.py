import pytest
import numpy as np
import cirq
from src.lobe.asp import (
    add_prepare_circuit,
    get_target_state,
)
from src.lobe._grover_rudolph import _ZERO


@pytest.mark.parametrize("number_of_qubits", range(1, 10))
@pytest.mark.parametrize(
    "number_of_terms", np.random.randint(2, (1 << 10) + 1, size=20)
)
@pytest.mark.parametrize("asp_func", [add_prepare_circuit])
def test_asp(number_of_qubits, number_of_terms, asp_func):
    number_of_terms = number_of_terms % (1 << number_of_qubits)
    if number_of_terms < 2:
        pytest.skip("Not enough terms")

    number_of_qubits = max(int(np.ceil(np.log2(number_of_terms))), 1)

    coefficients = np.random.uniform(2, 100, size=number_of_terms)
    target_state = get_target_state(coefficients)

    circuit = cirq.Circuit()
    qubits = [cirq.LineQubit(i) for i in range(number_of_qubits)]

    circuit.append(cirq.I.on_each(*qubits))
    gates, metrics = asp_func(qubits, target_state)
    circuit.append(gates)

    simulator = cirq.Simulator()
    wavefunction = simulator.simulate(circuit).final_state_vector
    for index, coeff in enumerate(target_state):
        assert np.isclose(wavefunction[index], coeff, atol=100 * _ZERO)

    assert metrics.number_of_rotations > 0
    assert metrics.number_of_rotations <= 1 << int(
        np.ceil(np.log2((len(coefficients))))
    )

    gates, _metrics = asp_func(qubits, target_state, dagger=True)
    metrics += _metrics
    circuit.append(gates)

    wavefunction = simulator.simulate(circuit).final_state_vector
    one_state = np.zeros(1 << number_of_qubits)
    one_state[0] = 1
    for index, coeff in enumerate(one_state):
        assert np.isclose(wavefunction[index], coeff, atol=100 * _ZERO)

    assert metrics.number_of_rotations > 0
    assert metrics.number_of_rotations <= 2 * (
        1 << int(np.ceil(np.log2((len(coefficients)))))
    )
