import cirq
import pytest
import random
import numpy as np
from functools import partial
from openparticle import ParticleOperator
from openparticle.hamiltonians.yukawa_hamiltonians import yukawa_hamiltonian
from src.lobe.asp import get_target_state, add_prepare_circuit
from src.lobe.index import index_over_terms
from src.lobe.metrics import CircuitMetrics
from src.lobe.rescale import rescale_coefficients
from src.lobe.system import System
from src.lobe.interaction import _determine_block_encoding_function
from src.lobe._utils import (
    translate_antifermions_to_fermions,
    get_bosonic_exponents,
)
from _utils import (
    _validate_block_encoding,
    _validate_block_encoding_does_nothing_when_control_is_off,
    _validate_clean_ancillae_are_cleaned,
    _validate_block_encoding_select_is_self_inverse,
    _make_be_func_self_inverse,
)


def _generate_operator(
    max_fermionic_mode,
    max_bosonic_mode,
    number_of_active_fermionic_modes,
    number_of_active_bosonic_modes,
    max_occupation,
):
    fermionic_indices = list(range(max_fermionic_mode + 1))
    random.shuffle(fermionic_indices)

    operator_string = ""
    for index in fermionic_indices[:number_of_active_fermionic_modes]:
        operator_string += (
            "b" + str(index) + np.random.choice(["^", ""], size=1)[0] + " "
        )

    bosonic_indices = list(range(max_bosonic_mode + 1))
    random.shuffle(bosonic_indices)
    for index in bosonic_indices[:number_of_active_bosonic_modes]:
        creation_exponent = np.random.choice(range(max_occupation))
        annihilation_exponent = np.random.choice(range(max_occupation))
        if (creation_exponent == 0) and (annihilation_exponent == 0):
            creation_exponent = 1
        for _ in range(creation_exponent):
            operator_string += "a" + str(index) + "^" + " "
        for _ in range(annihilation_exponent):
            operator_string += "a" + str(index) + " "

    return ParticleOperator(operator_string[:-1])


@pytest.mark.parametrize("number_of_inactive_fermionic_modes", [0, 1, 2])
@pytest.mark.parametrize("number_of_inactive_bosonic_modes", [0, 1, 2])
@pytest.mark.parametrize("number_of_active_fermionic_modes", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("number_of_active_bosonic_modes", [1, 2, 3])
@pytest.mark.parametrize("maximum_occupation_number", [1, 3])
@pytest.mark.parametrize("self_inverse", [True, False])
def test_interaction_terms(
    number_of_inactive_fermionic_modes,
    number_of_inactive_bosonic_modes,
    number_of_active_fermionic_modes,
    number_of_active_bosonic_modes,
    maximum_occupation_number,
    self_inverse,
):
    if (number_of_active_fermionic_modes == 0) and (
        number_of_active_bosonic_modes == 0
    ):
        number_of_active_bosonic_modes += 1
    number_of_fermionic_modes = (
        number_of_inactive_fermionic_modes + number_of_active_fermionic_modes
    )
    number_of_bosonic_modes = (
        number_of_inactive_bosonic_modes + number_of_active_bosonic_modes
    )
    term = _generate_operator(
        number_of_fermionic_modes - 1,
        number_of_bosonic_modes - 1,
        number_of_active_fermionic_modes,
        number_of_active_bosonic_modes,
        maximum_occupation_number,
    )
    if not term.is_hermitian:
        term += term.dagger()

    number_of_block_encoding_ancillae = number_of_active_bosonic_modes + 1
    if self_inverse:
        number_of_block_encoding_ancillae += 1

    ###############################################
    number_of_clean_ancillae = 100
    circuit = cirq.Circuit()
    control = cirq.LineQubit(0)
    block_encoding_ancillae = [
        cirq.LineQubit(i + 1) for i in range(number_of_block_encoding_ancillae)
    ]
    clean_ancillae = [
        cirq.LineQubit(i + 1 + number_of_block_encoding_ancillae)
        for i in range(number_of_clean_ancillae)
    ]
    number_of_fermionic_modes = 0
    number_of_bosonic_modes = 0
    if term.max_fermionic_mode is not None:
        number_of_fermionic_modes = term.max_fermionic_mode + 1
    if term.max_bosonic_mode is not None:
        number_of_bosonic_modes = term.max_bosonic_mode + 1
    system = System(
        maximum_occupation_number,
        1 + number_of_block_encoding_ancillae + number_of_clean_ancillae,
        number_of_fermionic_modes=number_of_fermionic_modes,
        number_of_bosonic_modes=number_of_bosonic_modes,
    )
    self_inverse_ancilla = None
    if self_inverse:
        self_inverse_ancilla = block_encoding_ancillae[0]
        block_encoding_ancillae = block_encoding_ancillae[1:]

    circuit.append(
        cirq.I.on_each(
            control,
            *block_encoding_ancillae,
            *system.fermionic_modes,
        )
    )
    for bosonic_reg in system.bosonic_modes:
        circuit.append(cirq.I.on_each(*bosonic_reg))
    # Flip control qubit so that we can focus on the 0-subspace of the control
    circuit.append(cirq.X.on(control))

    be_function, expected_rescaling_factor = _determine_block_encoding_function(
        term,
        system,
        block_encoding_ancillae,
        self_inverse_ancilla=self_inverse_ancilla,
        clean_ancillae=clean_ancillae,
    )
    if self_inverse:
        be_function = partial(
            _make_be_func_self_inverse,
            be_function=be_function,
            system=system,
            block_encoding_ancillae=block_encoding_ancillae,
            self_inverse_ancilla=self_inverse_ancilla,
            clean_ancillae=clean_ancillae,
        )
    gates, metrics = be_function(ctrls=([control], [1]))
    circuit += gates
    # Flip control qubit so that we can focus on the 0-subspace of the control
    circuit.append(cirq.X.on(control))
    #############################################################

    _validate_block_encoding(
        circuit,
        system,
        expected_rescaling_factor,
        term,
        number_of_block_encoding_ancillae,
        maximum_occupation_number,
    )
    _validate_clean_ancillae_are_cleaned(
        circuit, system, number_of_block_encoding_ancillae
    )
    _validate_block_encoding_does_nothing_when_control_is_off(
        circuit, system, number_of_block_encoding_ancillae
    )
    if self_inverse:
        _validate_block_encoding_select_is_self_inverse(
            circuit,
            system,
            term,
            number_of_block_encoding_ancillae,
            maximum_occupation_number,
        )

    rescaling_factor = 1
    bosonic_exponents = get_bosonic_exponents(term.to_list()[0])[1]
    for exponents in bosonic_exponents:
        rescaling_factor *= np.sqrt(maximum_occupation_number) ** (sum(exponents))
    W = np.ceil(np.log2(maximum_occupation_number + 1))
    assert np.isclose(expected_rescaling_factor, rescaling_factor)
    assert (
        metrics.number_of_t_gates + (4 * metrics.number_of_elbows)
        <= (number_of_active_bosonic_modes * (4 * (W + (2 * (W - 1)))))
        + 4 * (number_of_active_fermionic_modes - 1)
        + 4
    )
    assert metrics.number_of_nonclifford_rotations <= number_of_active_bosonic_modes * (
        maximum_occupation_number + 3
    )


@pytest.mark.parametrize("number_of_terms", [2, 4, 8, 16])
def test_full_yukawa(number_of_terms):

    resolution = 3
    operator = yukawa_hamiltonian(resolution, 1, 1, 1)
    translated_operator = translate_antifermions_to_fermions(operator).normal_order()
    groups = np.random.choice(translated_operator.group(), size=number_of_terms)

    replacement_operator = groups[0]
    for group in groups[1:]:
        replacement_operator += group

    number_of_block_encoding_ancillae = 3
    maximum_occupation_number = 1

    ###############################################
    ctrls = ([cirq.LineQubit(0)], [1])
    block_encoding_ancillae = [
        cirq.LineQubit(i + 1) for i in range(number_of_block_encoding_ancillae)
    ]
    index_register = [
        cirq.LineQubit(i + 100) for i in range(int(np.ceil(np.log2(len(groups)))))
    ]
    # index_register = [cirq.LineQubit(100)]
    clean_ancillae = [cirq.LineQubit(i + 200) for i in range(100)]
    number_of_fermionic_modes = 0
    number_of_bosonic_modes = 0
    if replacement_operator.max_fermionic_mode is not None:
        number_of_fermionic_modes = replacement_operator.max_fermionic_mode + 1
    if replacement_operator.max_bosonic_mode is not None:
        number_of_bosonic_modes = replacement_operator.max_bosonic_mode + 1
    system = System(
        maximum_occupation_number=maximum_occupation_number,
        number_of_used_qubits=1000,
        number_of_fermionic_modes=number_of_fermionic_modes,
        number_of_bosonic_modes=number_of_bosonic_modes,
    )

    block_encoding_functions = []
    rescaling_factors = []
    for term in replacement_operator.group():
        be_func, rescaling_factor = _determine_block_encoding_function(
            term,
            system,
            block_encoding_ancillae,
            clean_ancillae=clean_ancillae,
        )
        block_encoding_functions.append(be_func)
        rescaling_factors.append(rescaling_factor)

    rescaled_coefficients, overall_rescaling_factor = rescale_coefficients(
        [np.abs(group.coeffs[0]) for group in replacement_operator.group()],
        rescaling_factors,
    )
    target_state = get_target_state(rescaled_coefficients)

    # Generate Circuit
    gates = []
    gates.append(cirq.I.on_each(*system.fermionic_modes))
    for register in system.bosonic_modes:
        gates.append(cirq.I.on_each(*register))
    metrics = CircuitMetrics()

    gates.append(cirq.X.on(ctrls[0][0]))
    _gates, _metrics = add_prepare_circuit(
        index_register, target_state, clean_ancillae=clean_ancillae
    )
    gates += _gates
    metrics += _metrics

    _gates, _metrics = index_over_terms(
        index_register, block_encoding_functions, clean_ancillae, ctrls=ctrls
    )
    gates += _gates
    metrics += _metrics

    _gates, _metrics = add_prepare_circuit(
        index_register, target_state, dagger=True, clean_ancillae=clean_ancillae
    )
    gates += _gates
    metrics += _metrics
    gates.append(cirq.X.on(ctrls[0][0]))

    circuit = cirq.Circuit(gates)

    _validate_clean_ancillae_are_cleaned(
        circuit, system, len(index_register) + number_of_block_encoding_ancillae
    )
    _validate_block_encoding_does_nothing_when_control_is_off(
        circuit, system, len(index_register) + number_of_block_encoding_ancillae
    )
    _validate_block_encoding(
        circuit,
        system,
        overall_rescaling_factor,
        replacement_operator,
        len(index_register) + number_of_block_encoding_ancillae,
        maximum_occupation_number,
        max_qubits=22,
    )
