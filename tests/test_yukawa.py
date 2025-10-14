import cirq
import pytest
import numpy as np
from openparticle import ParticleOperator
from openparticle.hamiltonians.yukawa_hamiltonians import yukawa_hamiltonian
from openparticle.hamiltonians.renormalized_yukawa_hamiltonian import (
    renormalized_yukawa_hamiltonian,
)
from src.lobe.asp import get_target_state, add_prepare_circuit
from src.lobe.index import index_over_terms
from src.lobe.metrics import CircuitMetrics
from src.lobe.rescale import rescale_coefficients
from src.lobe.system import System
from src.lobe.yukawa import _determine_block_encoding_function
from src.lobe._utils import (
    translate_antifermions_to_fermions,
    predict_number_of_block_encoding_ancillae,
)
from _utils import (
    _validate_block_encoding,
    _validate_block_encoding_does_nothing_when_control_is_off,
    _validate_clean_ancillae_are_cleaned,
)


@pytest.mark.parametrize(
    "term",
    translate_antifermions_to_fermions(yukawa_hamiltonian(3, 1, 1, 1))
    .normal_order()
    .group(),
)
@pytest.mark.parametrize("reindex", [True])
def test_all_yukawa_terms(term, reindex):
    if reindex:
        indices = [op.max_mode for op in term.to_list()[0].split()]
        replacement_indices = []

        index_counter = 0
        for i, index in enumerate(indices):
            if index in indices[:i]:
                replacement_indices.append(replacement_indices[indices.index(index)])
            else:
                replacement_indices.append(index_counter)
                index_counter += 1

        new_term_string = ""
        for i, op in enumerate(term.to_list()[0].split()):
            op_key = list(op.op_dict.keys())[0][0]
            if op_key[0] == 0:
                new_term_string += "b"
            else:
                assert op_key[0] == 2
                new_term_string += "a"
            new_term_string += str(replacement_indices[i])

            if op_key[2]:
                new_term_string += "^"

            new_term_string += " "

        replacement_term = ParticleOperator(new_term_string)
        if len(term.to_list()) == 2:
            replacement_term += replacement_term.dagger()

        term = replacement_term

    number_of_block_encoding_ancillae = 3
    maximum_occupation_number = 3

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
        clean_ancillae=clean_ancillae,
    )
    gates, metrics = be_function(ctrls=([control], [1]))
    circuit += gates
    # Flip control qubit so that we can focus on the 0-subspace of the control
    circuit.append(cirq.X.on(control))
    #############################################################

    expected_rescaling_factor *= np.abs(term.coeffs[0])
    _validate_clean_ancillae_are_cleaned(
        circuit, system, number_of_block_encoding_ancillae
    )
    _validate_block_encoding_does_nothing_when_control_is_off(
        circuit, system, number_of_block_encoding_ancillae
    )
    _validate_block_encoding(
        circuit,
        system,
        expected_rescaling_factor,
        term,
        number_of_block_encoding_ancillae,
        maximum_occupation_number,
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
            term, system, block_encoding_ancillae, clean_ancillae=clean_ancillae
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


@pytest.mark.parametrize(
    "original_term",
    translate_antifermions_to_fermions(
        renormalized_yukawa_hamiltonian(res=3, t=1e-2, g=0.5, mf=1, mb=0.6)
    )
    .normal_order()
    .group(),
)
@pytest.mark.parametrize("reindex", [True])
def test_all_renorm_yukawa_terms(original_term, reindex):
    term = original_term
    if reindex:
        indices = [op.max_mode for op in original_term.to_list()[0].split()]
        replacement_indices = []

        index_counter = 0
        for i, index in enumerate(indices):
            if index in indices[:i]:
                replacement_indices.append(replacement_indices[indices.index(index)])
            else:
                replacement_indices.append(index_counter)
                index_counter += 1

        new_term_string = ""
        for i, op in enumerate(original_term.to_list()[0].split()):
            op_key = list(op.op_dict.keys())[0][0]
            if op_key[0] == 0:
                new_term_string += "b"
            else:
                assert op_key[0] == 2
                new_term_string += "a"
            new_term_string += str(replacement_indices[i])

            if op_key[2]:
                new_term_string += "^"

            new_term_string += " "

        replacement_term = ParticleOperator(new_term_string)
        if len(original_term.to_list()) == 2:
            replacement_term += replacement_term.dagger()

        term = replacement_term

    number_of_block_encoding_ancillae = predict_number_of_block_encoding_ancillae(term)
    maximum_occupation_number = 3

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
        clean_ancillae=clean_ancillae,
    )
    gates, metrics = be_function(ctrls=([control], [1]))
    circuit += gates
    # Flip control qubit so that we can focus on the 0-subspace of the control
    circuit.append(cirq.X.on(control))
    #############################################################

    expected_rescaling_factor *= np.abs(term.coeffs[0])
    _validate_clean_ancillae_are_cleaned(
        circuit, system, number_of_block_encoding_ancillae
    )
    _validate_block_encoding_does_nothing_when_control_is_off(
        circuit, system, number_of_block_encoding_ancillae
    )
    _validate_block_encoding(
        circuit,
        system,
        expected_rescaling_factor,
        term,
        number_of_block_encoding_ancillae,
        maximum_occupation_number,
    )


@pytest.mark.parametrize("number_of_terms", [2, 4, 8, 16])
def test_renormalized_yukawa(number_of_terms):
    # arbitrary model parameters
    g = 0.5
    mf = 1
    mb = 0.6

    # resolution
    K = 3

    # renormalization scale parameter
    t = 1e-2
    operator = renormalized_yukawa_hamiltonian(res=K, t=t, g=g, mf=mf, mb=mb)
    translated_operator = translate_antifermions_to_fermions(operator).normal_order()
    groups = np.random.choice(translated_operator.group(), size=number_of_terms)

    replacement_operator = groups[0]
    for group in groups[1:]:
        replacement_operator += group

    number_of_block_encoding_ancillae = max(
        [predict_number_of_block_encoding_ancillae(group) for group in groups]
    )
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
    coefficients = []
    for term in replacement_operator.group():
        term = term.mode_order()
        coefficients.append(np.abs(term.coeffs[0]))
        be_func, rescaling_factor = _determine_block_encoding_function(
            term, system, block_encoding_ancillae, clean_ancillae=clean_ancillae
        )
        block_encoding_functions.append(be_func)
        rescaling_factors.append(rescaling_factor)

    rescaled_coefficients, overall_rescaling_factor = rescale_coefficients(
        coefficients,
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
    #############################################################

    _validate_clean_ancillae_are_cleaned(
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
        using_pytest=False,
    )
