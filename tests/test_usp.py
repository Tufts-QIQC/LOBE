import cirq
import pytest
import numpy as np
from src.lobe.usp import diffusion_operator


@pytest.mark.parametrize("number_of_operators", range(1, 17))
def test_naive_usp_prepares_correct_state(number_of_operators):
    number_of_qubits = int(np.ceil(np.log2(number_of_operators)))
    simulator = cirq.Simulator(dtype=np.complex128)

    circuit = cirq.Circuit()
    qubits = [cirq.LineQubit(i) for i in range(number_of_qubits)]

    circuit.append(diffusion_operator(qubits))

    wavefunction = simulator.simulate(circuit).final_state_vector

    expected_wavefunction = (1 / np.sqrt(1 << number_of_qubits)) * np.ones(
        1 << number_of_qubits
    )

    assert np.allclose(expected_wavefunction, wavefunction)
