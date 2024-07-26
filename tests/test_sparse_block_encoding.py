import pytest
import cirq
import numpy as np
from src.lobe.usp import add_naive_usp
from src.lobe.coefficient_oracle import add_coefficient_oracle
from src.lobe.select_oracle import add_select_oracle
from src.lobe.system import System
from openparticle import ParticleOperator


@pytest.mark.parametrize(
    ["operators", "coefficients", "hamiltonian"],
    [
        (
            (
                ParticleOperator("b0^ b0")
                + ParticleOperator("b0^ b1")
                + ParticleOperator("b1^ b0")
                + ParticleOperator("b1^ b1")
            ),
            np.array([1, 1, 1, 1]),
            np.array(
                [
                    np.array([0, 0, 0, 0]),
                    np.array([0, 1, 1, 0]),
                    np.array([0, 1, 1, 0]),
                    np.array([0, 0, 0, 2]),
                ]
            ),
        ),
        (
            (
                ParticleOperator("b0^ b0")
                + ParticleOperator("b0^ b1")
                + ParticleOperator("b1^ b0")
                + ParticleOperator("b1^ b1")
            ),
            np.array([1, 0.5, 0.5, 1]),
            np.array(
                [
                    np.array([0, 0, 0, 0]),
                    np.array([0, 1, 0.5, 0]),
                    np.array([0, 0.5, 1, 0]),
                    np.array([0, 0, 0, 2]),
                ]
            ),
        ),
        (
            (
                ParticleOperator("b0^ b0")
                + ParticleOperator("b1^ b1")
                + ParticleOperator("b2^ b2")
            ),
            np.array([4, 2, 1]),
            np.array(
                [
                    np.array([0, 0, 0, 0, 0, 0, 0, 0]),
                    np.array([0, 4, 0, 0, 0, 0, 0, 0]),
                    np.array([0, 0, 2, 0, 0, 0, 0, 0]),
                    np.array([0, 0, 0, 6, 0, 0, 0, 0]),
                    np.array([0, 0, 0, 0, 1, 0, 0, 0]),
                    np.array([0, 0, 0, 0, 0, 5, 0, 0]),
                    np.array([0, 0, 0, 0, 0, 0, 3, 0]),
                    np.array([0, 0, 0, 0, 0, 0, 0, 7]),
                ]
            ),
        ),
        (
            (
                ParticleOperator("b0^ b0")
                + ParticleOperator("b0^ b1")
                + ParticleOperator("b0^ b2")
                + ParticleOperator("b0^ b3")
                + ParticleOperator("b1^ b0")
                + ParticleOperator("b1^ b1")
                + ParticleOperator("b1^ b2")
                + ParticleOperator("b1^ b3")
                + ParticleOperator("b2^ b0")
                + ParticleOperator("b2^ b1")
                + ParticleOperator("b2^ b2")
                + ParticleOperator("b2^ b3")
                + ParticleOperator("b3^ b0")
                + ParticleOperator("b3^ b1")
                + ParticleOperator("b3^ b2")
                + ParticleOperator("b3^ b3")
            ),
            np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]),
            np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 2, 0, 3, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0],
                    [0, 5, 6, 0, 7, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 7, 0, 7, -3, 0, 0, 8, -4, 0, 0, 0, 0, 0],
                    [0, 9, 10, 0, 11, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 10, 0, 12, 2, 0, 0, 12, 0, 0, -4, 0, 0, 0],
                    [0, 0, 0, -9, 0, 5, 17, 0, 0, 0, 12, 0, -8, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 18, 0, 0, 0, 12, 0, -8, 4, 0],
                    [0, 13, 14, 0, 15, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 14, 0, 15, 0, 0, 0, 17, 2, 0, 3, 0, 0, 0],
                    [0, 0, 0, -13, 0, 0, 15, 0, 0, 5, 22, 0, 7, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 15, 0, 0, 0, 23, 0, 7, -3, 0],
                    [0, 0, 0, 0, 0, -13, -14, 0, 0, 9, 10, 0, 27, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, -14, 0, 0, 0, 10, 0, 28, 2, 0],
                    [0, 0, 0, 0, 0, 0, 0, 13, 0, 0, 0, -9, 0, 5, 33, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 34],
                ]
            ),
        ),
        (
            (
                ParticleOperator("b0^ b2")
            ),  # Operator b^\dagger_0 b_2. Should map |100> -> |001> and |110> -> -|011>
            np.array([1]),
            np.array(
                [
                    np.array([0, 0, 0, 0, 0, 0, 0, 0]),
                    np.array([0, 0, 0, 0, 1, 0, 0, 0]),
                    np.array([0, 0, 0, 0, 0, 0, 0, 0]),
                    np.array([0, 0, 0, 0, 0, 0, -1, 0]),
                    np.array([0, 0, 0, 0, 0, 0, 0, 0]),
                    np.array([0, 0, 0, 0, 0, 0, 0, 0]),
                    np.array([0, 0, 0, 0, 0, 0, 0, 0]),
                    np.array([0, 0, 0, 0, 0, 0, 0, 0]),
                ]
            ),
        ),
    ],
)
def test_block_encoding_for_toy_hamiltonian(operators, coefficients, hamiltonian):
    number_of_index_qubits = max(int(np.ceil(np.log2(len(operators.to_list())))), 1)
    circuit = cirq.Circuit()
    validation = cirq.LineQubit(0)
    clean_ancilla = [cirq.LineQubit(1)]
    rotation = cirq.LineQubit(2)
    index = [cirq.LineQubit(i + 3) for i in range(number_of_index_qubits)]
    bosonic_rotation_register = []
    system = System(
        # number_of_modes=max(modes) + 1,
        number_of_modes=operators.max_mode() + 1,
        number_of_used_qubits=3 + number_of_index_qubits,
        has_fermions=True,
    )
    normalization_factor = max(coefficients)
    normalized_coefficients = coefficients / normalization_factor

    circuit = add_naive_usp(circuit, index)
    circuit.append(cirq.X.on(validation))
    circuit = add_select_oracle(
        circuit,
        validation,
        index,
        system,
        operators,
        bosonic_rotation_register,
        clean_ancilla,
    )
    circuit = add_coefficient_oracle(
        circuit, rotation, index, normalized_coefficients, len(operators.to_list())
    )
    circuit = add_naive_usp(circuit, index)

    upper_left_block = circuit.unitary()[
        : 1 << system.number_of_system_qubits, : 1 << system.number_of_system_qubits
    ]
    normalized_hamiltonian = hamiltonian / (
        (1 << number_of_index_qubits) * normalization_factor
    )
    assert np.allclose(upper_left_block, normalized_hamiltonian)


def test_select_and_coefficient_oracles_commute():
    operators = (
        ParticleOperator("b0^ b0")
        + ParticleOperator("b0^ b1")
        + ParticleOperator("b1^ b0")
        + ParticleOperator("b1^ b1")
    ).to_list()
    coefficients = np.random.uniform(0, 1, size=4)
    validation = cirq.LineQubit(0)
    rotation = cirq.LineQubit(1)
    index = [cirq.LineQubit(i + 2) for i in range(2)]
    clean_ancilla = [cirq.LineQubit(4)]
    bosonic_rotation_register = []
    system = System(
        number_of_modes=2,
        number_of_used_qubits=5,
        has_fermions=True,
    )

    circuit = cirq.Circuit()
    circuit = add_naive_usp(circuit, index)
    circuit.append(cirq.X.on(validation))
    circuit = add_select_oracle(
        circuit,
        validation,
        index,
        system,
        operators,
        bosonic_rotation_register,
        clean_ancilla,
    )
    circuit = add_coefficient_oracle(
        circuit, rotation, index, coefficients, len(operators)
    )
    circuit = add_naive_usp(circuit, index)
    unitary_select_first = circuit.unitary()

    circuit = cirq.Circuit()
    circuit = add_naive_usp(circuit, index)
    circuit.append(cirq.X.on(validation))
    circuit = add_coefficient_oracle(
        circuit, rotation, index, coefficients, len(operators)
    )
    circuit = add_select_oracle(
        circuit,
        validation,
        index,
        system,
        operators,
        bosonic_rotation_register,
        clean_ancilla,
    )
    circuit = add_naive_usp(circuit, index)
    unitary_coefficient_first = circuit.unitary()

    assert np.allclose(unitary_select_first, unitary_coefficient_first)
