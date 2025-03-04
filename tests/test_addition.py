import cirq
import pytest
import numpy as np
from src.lobe.addition import (
    add_incrementer,
    add_classical_value_incrementers,
    add_classical_value_gate_efficient,
    add_classical_value,
)


@pytest.mark.parametrize("number_of_qubits", range(1, 10))
@pytest.mark.parametrize("integer", np.random.randint(1, (1 << 10) + 1, size=10))
@pytest.mark.parametrize("decrement", [True, False])
def test_incrementer_on_basis_state(number_of_qubits, integer, decrement):
    integer = integer % (1 << number_of_qubits)
    # Create a quantum circuit
    circuit = cirq.Circuit()

    # Create qubits
    ancilla = [cirq.LineQubit(i) for i in range(100)]
    qubits = [cirq.LineQubit(i + 100) for i in range(number_of_qubits)]
    circuit = cirq.Circuit()
    circuit.append(cirq.I.on_each(*qubits))
    _gates, _ = add_incrementer(qubits, ancilla, decrement=decrement)
    circuit += _gates

    initial_state = np.zeros(1 << number_of_qubits)
    initial_state[integer] = 1
    initial_ancilla_state = np.zeros(
        1 << (len(circuit.all_qubits()) - number_of_qubits)
    )
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


@pytest.mark.parametrize("number_of_qubits", range(1, 10))
@pytest.mark.parametrize("integer_one", np.random.randint(1, (1 << 10) + 1, size=5))
@pytest.mark.parametrize("integer_two", np.random.randint(1, (1 << 10) + 1, size=5))
@pytest.mark.parametrize("decrement", [True, False])
def test_incrementer_on_superposition_state(
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

    circuit = cirq.Circuit()
    _gates, _ = add_incrementer(qubits, ancilla, decrement=decrement)
    circuit.append(_gates)

    initial_state = np.zeros(1 << number_of_qubits, dtype=np.complex128)
    initial_state[integer_one] += random_amplitudes[0]
    initial_state[integer_two] += random_amplitudes[1]
    initial_ancilla_state = np.zeros(
        1 << (len(circuit.all_qubits()) - number_of_qubits)
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


@pytest.mark.parametrize("number_of_qubits", range(1, 10))
@pytest.mark.parametrize("integer", np.random.randint(1, (1 << 10) + 1, size=10))
@pytest.mark.parametrize("control_value", [0, 1])
@pytest.mark.parametrize("decrement", [True, False])
def test_incrementer_is_properly_controlled(
    number_of_qubits, integer, control_value, decrement
):
    integer = integer % (1 << number_of_qubits)
    # Create a quantum circuit
    circuit = cirq.Circuit()

    # Create qubits
    control = cirq.LineQubit(0)
    ancilla = [cirq.LineQubit(i + 1) for i in range(100)]
    qubits = [cirq.LineQubit(i + 101) for i in range(number_of_qubits)]

    circuit = cirq.Circuit()
    _gates, _ = add_incrementer(
        qubits, ancilla, decrement=decrement, ctrls=([control], [1])
    )
    circuit.append(_gates)

    initial_control_state = np.zeros(2)
    initial_control_state[control_value] = 1
    initial_ancilla_state = np.zeros(
        1 << (len(circuit.all_qubits()) - 1 - number_of_qubits)
    )
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
        if decrement:
            expected_integer = integer - 1
        else:
            expected_integer = integer + 1
        expected_integer = (expected_integer % (1 << number_of_qubits)) + (
            1 << (len(circuit.all_qubits()) - 1)
        )

    assert final_state[expected_integer] == 1


@pytest.mark.parametrize(
    "add_classical_value_function",
    [
        add_classical_value_incrementers,
        add_classical_value_gate_efficient,
        add_classical_value,
    ],
)
@pytest.mark.parametrize("number_of_qubits", range(1, 10))
@pytest.mark.parametrize("integer", np.random.randint(1, (1 << 9) + 1, size=5))
@pytest.mark.parametrize("classical_value", np.random.randint(1, (1 << 9) + 1, size=5))
@pytest.mark.parametrize("decrement", [False])
@pytest.mark.parametrize("is_controlled", [True, False])
def test_add_classical_value_on_basis_state(
    add_classical_value_function,
    number_of_qubits,
    integer,
    classical_value,
    decrement,
    is_controlled,
):
    integer = integer % (1 << number_of_qubits)
    if decrement:
        classical_value *= -1

    if not is_controlled:
        expected_integer = (integer + classical_value) % (1 << number_of_qubits)
    else:
        expected_integer = integer

    # Create a quantum circuit
    circuit = cirq.Circuit()

    # Create qubits
    ctrls = ([], [])
    indexor = 0
    if is_controlled:
        ctrls = ([cirq.LineQubit(0)], [1])
        indexor += 1
    ancilla = [cirq.LineQubit(i + 100 + indexor) for i in range(100)]
    indexor += 100
    qubits = [cirq.LineQubit(i + 200 + indexor) for i in range(number_of_qubits)]
    circuit.append(cirq.I.on_each(*ctrls[0]))
    circuit.append(cirq.I.on_each(*qubits))
    _gates, _ = add_classical_value_function(
        qubits, classical_value, ancilla, ctrls=ctrls
    )
    circuit.append(_gates)

    initial_state = np.zeros(1 << number_of_qubits)
    initial_state[integer] = 1
    initial_ancilla_state = np.zeros(
        1 << len(circuit.all_qubits()) - len(ctrls[0]) - len(qubits)
    )
    initial_ancilla_state[0] = 1
    initial_state = np.kron(initial_ancilla_state, initial_state)
    if is_controlled:
        initial_control_state = np.zeros(2)
        initial_control_state[0] = 1
        initial_state = np.kron(initial_control_state, initial_state)

    if len(circuit.all_qubits()) > 20:
        pytest.skip("Too many qubits to simulate")
    simulator = cirq.Simulator()

    final_state = simulator.simulate(
        circuit, initial_state=initial_state
    ).final_state_vector

    assert final_state[expected_integer] == 1


@pytest.mark.parametrize("number_of_qubits", range(1, 20))
@pytest.mark.parametrize("number_of_zero_bits", range(1, 20))
def test_gate_efficient_adder_numerics(number_of_qubits, number_of_zero_bits):
    number_of_zero_bits = number_of_zero_bits % (number_of_qubits)
    number_to_add = 0
    for i in range(number_of_qubits):
        if i == number_of_zero_bits:
            number_to_add += 1 << i
        elif i >= number_of_zero_bits:
            if np.random.choice([True, False], 1):
                number_to_add += 1 << i

    qubits = [cirq.LineQubit(i) for i in range(number_of_qubits)]
    clean_ancillae = [cirq.LineQubit(i + 100 + number_of_qubits) for i in range(100)]
    ctrls = ([cirq.LineQubit(200 + number_of_qubits)], [1])

    _, metrics = add_classical_value_gate_efficient(
        qubits, number_to_add, clean_ancillae, ctrls=ctrls
    )

    assert metrics.number_of_elbows == number_of_qubits - number_of_zero_bits - 1
    assert metrics.clean_ancillae_usage[-1] == 0
    assert (
        max(metrics.clean_ancillae_usage)
        == (2 * (number_of_qubits - number_of_zero_bits)) - 1
    )


@pytest.mark.parametrize("number_of_qubits", range(1, 20))
def test_incrementer_numerics(number_of_qubits):
    qubits = [cirq.LineQubit(i) for i in range(number_of_qubits)]
    clean_ancillae = [cirq.LineQubit(i + 100 + number_of_qubits) for i in range(100)]
    ctrls = ([cirq.LineQubit(200 + number_of_qubits)], [1])

    _, metrics = add_incrementer(qubits, clean_ancillae, ctrls=ctrls)

    if number_of_qubits <= 1:
        assert metrics.number_of_elbows == 0
        assert metrics.clean_ancillae_usage == []
    else:
        assert metrics.number_of_elbows == number_of_qubits - 1
        if len(metrics.clean_ancillae_usage) > 0:
            assert metrics.clean_ancillae_usage[-1] == 0
            assert max(metrics.clean_ancillae_usage) == number_of_qubits - 1

    _, metrics = add_incrementer(qubits, clean_ancillae)

    if number_of_qubits <= 2:
        assert metrics.number_of_elbows == 0
        assert metrics.clean_ancillae_usage == []
    else:
        assert metrics.number_of_elbows == number_of_qubits - 2
        if len(metrics.clean_ancillae_usage) > 0:
            assert metrics.clean_ancillae_usage[-1] == 0
            assert max(metrics.clean_ancillae_usage) == number_of_qubits - 2
