import numpy as np
import pytest
import cirq
from src.lobe.system import System
from src.lobe.rescale import (
    get_numbers_of_bosonic_operators_in_terms,
    bosonically_rescale_terms,
)
from src.lobe.asp import get_target_state
from src.lobe.block_encoding import add_lobe_oracle
from src.lobe.qubitization import _add_reflection, add_qubitized_walk_operator
from openparticle import ParticleOperator


def _get_operator():
    possible_types = [
        ["fermion"],
        ["antifermion"],
        ["boson"],
        ["fermion", "antifermion"],
        ["fermion", "boson"],
        ["antifermion", "boson"],
        ["fermion", "antifermion", "boson"],
    ]
    types = possible_types[np.random.choice(range(7))]
    n_terms = np.random.choice(range(1, 17))
    max_mode = np.random.choice(range(1, 9))
    max_len_of_terms = np.random.choice(range(1, 9))
    operator = ParticleOperator.random(
        types,
        n_terms,
        max_mode=max_mode,
        max_len_of_terms=max_len_of_terms,
        complex_coeffs=False,
        normal_order=True,
    ).normal_order()
    operator.remove_identity()

    terms = operator.to_list()

    operator = terms[0]
    for term in terms[1:]:
        operator += term

    maximum_occupation_number = np.random.choice([1, 3, 7])

    rescaled_terms, _ = bosonically_rescale_terms(terms, maximum_occupation_number)
    rescaled_coefficients = [term.coeff for term in rescaled_terms]

    if len(rescaled_coefficients) == 1:
        return _get_operator()
    return rescaled_terms, rescaled_coefficients, operator


@pytest.mark.parametrize("trial", range(10))
def test_reflection_operator(trial):
    terms, rescaled_coefficients, _ = _get_operator()
    norm = sum(np.abs(rescaled_coefficients))
    target_state = get_target_state(rescaled_coefficients)

    num_index_qubits = int(np.ceil(np.log2(len(terms))))
    prepared_state = np.zeros(1 << num_index_qubits)
    for i, coefficient in enumerate(rescaled_coefficients):
        prepared_state[i] = np.sqrt(np.abs(coefficient) / norm)

    number_of_index_qubits = max(int(np.ceil(np.log2(len(terms)))), 1)
    number_of_rotation_qubits = (
        max(get_numbers_of_bosonic_operators_in_terms(terms)) + 1
    )

    # Declare Qubits
    reflection_circuit = cirq.Circuit()
    validation = cirq.LineQubit(0)
    rotation_qubits = [cirq.LineQubit(i + 1) for i in range(number_of_rotation_qubits)]
    index_register = [
        cirq.LineQubit(i + 1 + number_of_rotation_qubits)
        for i in range(number_of_index_qubits)
    ]
    reflection_circuit.append(
        cirq.I.on_each(
            validation,
            *rotation_qubits,
            *index_register,
        )
    )

    reflection_circuit += _add_reflection(target_state, index_register)

    if len(reflection_circuit.all_qubits()) >= 14:
        pytest.skip(
            f"too many qubits {len(reflection_circuit.all_qubits())} to explicitly validate"
        )

    reflection_op = reflection_circuit.unitary()

    expected_reflection_op = np.kron(
        np.eye(1 << (1 + number_of_rotation_qubits)),
        np.eye(1 << number_of_index_qubits)
        - (2 * np.outer(prepared_state, prepared_state)),
    )

    assert np.allclose(expected_reflection_op, reflection_op, atol=1e-7)


@pytest.mark.parametrize("trial", range(30))
def test_walk_operator(trial):
    terms, rescaled_coefficients, operator = _get_operator()

    norm = sum(np.abs(rescaled_coefficients))

    num_index_qubits = int(np.ceil(np.log2(len(terms))))
    prepared_state = np.zeros(1 << num_index_qubits)
    for i, coefficient in enumerate(rescaled_coefficients):
        prepared_state[i] = np.sqrt(np.abs(coefficient) / norm)

    number_of_ancillae = 100
    number_of_index_qubits = max(int(np.ceil(np.log2(len(terms)))), 1)
    number_of_rotation_qubits = (
        max(get_numbers_of_bosonic_operators_in_terms(terms)) + 1
    )

    # Declare Qubits
    select_circuit = cirq.Circuit()
    validation = cirq.LineQubit(0)
    clean_ancillae = [cirq.LineQubit(i + 1) for i in range(number_of_ancillae)]
    rotation_qubits = [
        cirq.LineQubit(i + 1 + number_of_ancillae)
        for i in range(number_of_rotation_qubits)
    ]
    index_register = [
        cirq.LineQubit(i + 1 + number_of_ancillae + number_of_rotation_qubits)
        for i in range(number_of_index_qubits)
    ]
    number_of_modes = max([term.max_mode() for term in terms]) + 1
    maximum_occupation_number = np.random.choice([1, 3, 7])
    system = System(
        number_of_modes=number_of_modes,
        maximum_occupation_number=maximum_occupation_number,
        number_of_used_qubits=1
        + number_of_ancillae
        + number_of_rotation_qubits
        + number_of_index_qubits,
        has_fermions=operator.has_fermions,
        has_antifermions=operator.has_antifermions,
        has_bosons=operator.has_bosons,
    )
    select_circuit.append(
        cirq.I.on_each(
            validation,
            *rotation_qubits,
            *index_register,
            *system.fermionic_register,
            *system.antifermionic_register,
        )
    )
    for bosonic_reg in system.bosonic_system:
        select_circuit.append(cirq.I.on_each(*bosonic_reg))

    select_circuit.append(
        cirq.I.on_each(
            validation,
            *rotation_qubits,
            *index_register,
        )
    )

    select_circuit += add_lobe_oracle(
        terms,
        validation,
        index_register,
        system,
        rotation_qubits,
        clean_ancillae,
        perform_coefficient_oracle=False,
        decompose=True,
    )
    if len(select_circuit.all_qubits()) >= 14:
        pytest.skip(
            f"too many qubits {len(select_circuit.all_qubits())} to explicitly validate"
        )
    V_op = select_circuit.unitary()

    walk_operator_circuit = cirq.Circuit()
    walk_operator_circuit.append(
        cirq.I.on_each(
            validation,
            *rotation_qubits,
            *index_register,
            *system.fermionic_register,
            *system.antifermionic_register,
        )
    )
    for bosonic_reg in system.bosonic_system:
        walk_operator_circuit.append(cirq.I.on_each(*bosonic_reg))

    walk_operator_circuit.append(
        cirq.I.on_each(
            validation,
            *rotation_qubits,
            *index_register,
        )
    )

    walk_operator_circuit += add_qubitized_walk_operator(
        terms,
        rescaled_coefficients,
        validation,
        clean_ancillae,
        rotation_qubits,
        index_register,
        system,
    )

    if len(walk_operator_circuit.all_qubits()) >= 14:
        pytest.skip(
            f"too many qubits {len(walk_operator_circuit.all_qubits())} to explicitly validate"
        )

    walk_operator_unitary = walk_operator_circuit.unitary()

    number_of_used_ancillae = len(walk_operator_circuit.all_qubits())
    number_of_used_ancillae -= (
        1
        + number_of_rotation_qubits
        + number_of_index_qubits
        + system.number_of_system_qubits
    )
    expected_S_op = np.kron(
        np.kron(
            np.eye(1 << (1 + number_of_used_ancillae + number_of_rotation_qubits)),
            np.eye(1 << number_of_index_qubits)
            - (2 * np.outer(prepared_state, prepared_state)),
        ),
        np.eye(1 << system.number_of_system_qubits),
    )

    expected_walk_operator = expected_S_op @ V_op * np.exp(1j * np.pi)

    assert np.allclose(expected_walk_operator, walk_operator_unitary, atol=1e-7)
