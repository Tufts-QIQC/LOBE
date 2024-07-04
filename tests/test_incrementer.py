import pytest
import cirq
import numpy as np
from src.lobe.incrementer import add_incrementer


@pytest.mark.parametrize(
    "number_of_qubits", [1, 2] + np.random.randint(3, 10 + 1, size=3).tolist()
)
@pytest.mark.parametrize("integer", np.random.randint(1, (1 << 10) + 1, size=3))
@pytest.mark.parametrize("decrement", [True, False])
def test_binary_incrementer_on_basis_state(number_of_qubits, integer, decrement):
    integer = integer % (1 << number_of_qubits)
    # Create a quantum circuit
    circuit = cirq.Circuit()

    # Create qubits
    ancilla = None
    if number_of_qubits > 2:
        ancilla = [cirq.LineQubit(i) for i in range(number_of_qubits - 2)]
    qubits = [cirq.LineQubit(i + number_of_qubits - 2) for i in range(number_of_qubits)]

    circuit = add_incrementer(circuit, qubits, ancilla, decrement=decrement)

    initial_state = np.zeros(1 << number_of_qubits)
    initial_state[integer] = 1
    if number_of_qubits > 2:
        initial_ancilla_state = np.zeros(1 << (number_of_qubits - 2))
        initial_ancilla_state[0] = 1
        initial_state = np.kron(initial_ancilla_state, initial_state)

    simulator = cirq.Simulator()

    final_state = simulator.simulate(
        circuit, initial_state=initial_state
    ).final_state_vector

    expected_integer = (integer + 1) % (1 << number_of_qubits)
    if decrement:
        expected_integer = (integer - 1) % (1 << number_of_qubits)

    assert final_state[expected_integer] == 1


@pytest.mark.parametrize(
    "number_of_qubits", [1, 2] + np.random.randint(3, 10 + 1, size=3).tolist()
)
@pytest.mark.parametrize("integer_one", np.random.randint(1, (1 << 10) + 1, size=3))
@pytest.mark.parametrize("integer_two", np.random.randint(1, (1 << 10) + 1, size=3))
@pytest.mark.parametrize("decrement", [True, False])
def test_binary_incrementer_on_superposition_state(
    number_of_qubits, integer_one, integer_two, decrement
):
    integer_one = integer_one % (1 << number_of_qubits)
    integer_two = integer_two % (1 << number_of_qubits)

    # Create a quantum circuit
    circuit = cirq.Circuit()

    # Create qubits
    ancilla = None
    if number_of_qubits > 2:
        ancilla = [cirq.LineQubit(i) for i in range(number_of_qubits - 2)]
    qubits = [cirq.LineQubit(i + number_of_qubits - 2) for i in range(number_of_qubits)]

    random_amplitudes = (
        np.random.uniform(-1, 1, size=2) + np.random.uniform(-1, 1, size=2) * 1j
    )
    random_amplitudes /= np.linalg.norm(random_amplitudes)

    circuit = add_incrementer(circuit, qubits, ancilla, decrement=decrement)

    initial_state = np.zeros(1 << number_of_qubits, dtype=np.complex128)
    initial_state[integer_one] += random_amplitudes[0]
    initial_state[integer_two] += random_amplitudes[1]
    if number_of_qubits > 2:
        initial_ancilla_state = np.zeros(
            1 << (number_of_qubits - 2), dtype=np.complex128
        )
        initial_ancilla_state[0] = 1
        initial_state = np.kron(initial_ancilla_state, initial_state)

    simulator = cirq.Simulator()

    final_state = simulator.simulate(
        circuit, initial_state=initial_state
    ).final_state_vector

    expected_integer_one = (integer_one + 1) % (1 << number_of_qubits)
    expected_integer_two = (integer_two + 1) % (1 << number_of_qubits)
    if decrement:
        expected_integer_one = (integer_one - 1) % (1 << number_of_qubits)
        expected_integer_two = (integer_two - 1) % (1 << number_of_qubits)

    if integer_one == integer_two:
        assert np.isclose(final_state[expected_integer_one], sum(random_amplitudes))
    else:
        assert np.isclose(final_state[expected_integer_one], random_amplitudes[0])
        assert np.isclose(final_state[expected_integer_two], random_amplitudes[1])


@pytest.mark.parametrize(
    "number_of_qubits", [1, 2] + np.random.randint(3, 10 + 1, size=3).tolist()
)
@pytest.mark.parametrize("integer", np.random.randint(1, (1 << 10) + 1, size=3))
@pytest.mark.parametrize("control_value", [0, 1])
def test_binary_incrementer_is_properly_controlled(
    number_of_qubits, integer, control_value
):
    integer = integer % (1 << number_of_qubits)
    # Create a quantum circuit
    circuit = cirq.Circuit()

    # Create qubits
    control = [cirq.LineQubit(0)]
    qubit_counter = 1
    ancilla = None
    if number_of_qubits > 2:
        ancilla = [
            cirq.LineQubit(i + qubit_counter) for i in range(number_of_qubits - 2)
        ]
        qubit_counter += number_of_qubits - 2
    qubits = [cirq.LineQubit(i + qubit_counter) for i in range(number_of_qubits)]

    circuit = add_incrementer(circuit, qubits, ancilla, control_register=control)

    initial_control_state = np.zeros(2)
    initial_control_state[control_value] = 1
    if number_of_qubits > 2:
        initial_ancilla_state = np.zeros(1 << (number_of_qubits - 2))
        initial_ancilla_state[0] = 1
        initial_control_state = np.kron(initial_control_state, initial_ancilla_state)
    initial_state_system_state = np.zeros(1 << number_of_qubits)
    initial_state_system_state[integer] = 1
    initial_state = np.kron(
        initial_control_state,
        initial_state_system_state,
    )

    simulator = cirq.Simulator()

    final_state = simulator.simulate(
        circuit, initial_state=initial_state
    ).final_state_vector

    expected_integer = integer
    if control_value:
        expected_integer = ((integer + 1) % (1 << number_of_qubits)) + (
            1 << (len(circuit.all_qubits()) - 1)
        )

    assert final_state[expected_integer] == 1
