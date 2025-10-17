import cirq
import pytest
import numpy as np
from functools import partial
from openparticle import ParticleOperator, generate_matrix
from openparticle.hamiltonians.yukawa_hamiltonians import yukawa_hamiltonian

import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath('__file__')), '../'))
from src.lobe.system import System
from src.lobe.asp import add_prepare_circuit, get_target_state
from _utils import get_basis_of_full_system
from src.lobe._utils import translate_antifermions_to_fermions
from src.lobe.fermionic import fermionic_product_block_encoding, fermionic_plus_hc_block_encoding
from src.lobe.multiplexed_rotations import get_decomposed_multiplexed_rotation_circuit
from src.lobe._utils import _apply_negative_identity, get_basis_of_full_system, get_fermionic_operator_types, pretty_print
from src.lobe.bosonic import (
    _get_bosonic_rotation_angles, _add_multi_bosonic_rotations,
    self_inverse_bosonic_number_operator_block_encoding, self_inverse_bosonic_product_plus_hc_block_encoding)
from src.lobe.addition import add_classical_value
from src.lobe.decompose import decompose_controls_left, decompose_controls_right
from src.lobe.index import index_over_terms
from src.lobe.metrics import CircuitMetrics
from tests._utils import (
    _setup,
    _validate_block_encoding,
    _validate_block_encoding_does_nothing_when_control_is_off,
    _validate_block_encoding_select_is_self_inverse,
    _validate_clean_ancillae_are_cleaned,
)
from src.lobe.yukawa import _determine_block_encoding_function

def _toy_fermion_boson_interaction_be_term(
        system,
        block_encoding_ancillae,
        active_boson_modes,
        active_fermion_modes,
        bosonic_exponents_list,
        fermion_operator_types,
        clean_ancillae,
        ctrls = ([], [])
):
    

    unitary_index = block_encoding_ancillae[0]
    fermion_be_anc = block_encoding_ancillae[1]
    rotation_qubit = block_encoding_ancillae[2]
    
    gates = []
    metrics = CircuitMetrics()

    gates.append(cirq.H.on(unitary_index))
    gates.append(cirq.X.on(rotation_qubit).controlled_by(unitary_index))

    left_elbow, _metrics = decompose_controls_left(
        ([ctrls[0][0], system.fermionic_modes[0]], [1, 0]), clean_ancillae[0]
    )
    gates += left_elbow
    metrics += _metrics

    plus_one, _metrics = add_classical_value(system.bosonic_modes[active_boson_modes[0]], +1, clean_ancillae=clean_ancillae[1:], ctrls = ([clean_ancillae[0]], [1]))
    gates.append(plus_one)
    metrics += _metrics

    gates.append(cirq.X.on(clean_ancillae[0]).controlled_by(ctrls[0][0]))#left elbow followed by right elbow is CNOT

    rotation_gates, _metrics = _add_multi_bosonic_rotations(
                rotation_qubit,
                system.bosonic_modes[active_boson_modes[0]],
                bosonic_exponents_list[0][0],
                bosonic_exponents_list[0][1],
                clean_ancillae=clean_ancillae[1:],
                ctrls=ctrls,
            )
    gates.append(rotation_gates)
    metrics += _metrics

    minus_one, _metrics = add_classical_value(system.bosonic_modes[active_boson_modes[0]], -1, clean_ancillae=clean_ancillae[1:], ctrls = ([clean_ancillae[0]], [1]))
    gates.append(minus_one)
    metrics += _metrics

    right_elbow, _metrics = decompose_controls_right(
        ([ctrls[0][0], system.fermionic_modes[0]], [1, 1]), clean_ancillae[0]
    )
    gates += right_elbow
    metrics += _metrics

    gates.append(cirq.X.on(rotation_qubit).controlled_by(unitary_index))
    gates.append(cirq.X.on(unitary_index))
    gates.append(cirq.H.on(unitary_index))


    _gates, _metrics = fermionic_plus_hc_block_encoding(
                    system=system,
                    block_encoding_ancillae=[fermion_be_anc],
                    active_indices=active_fermion_modes,
                    operator_types=fermion_operator_types,
                    sign = 1,
                    clean_ancillae=clean_ancillae,
                    ctrls = ctrls
                )
    gates += _gates
    metrics += _metrics
    
    return gates, metrics


@pytest.mark.parametrize('maximum_occupation_number', [1, 3])
def test_fermionic_self_inverse_bosonic_product_plus_hc_satisfies_qubitization_conditions(maximum_occupation_number):
    """
    Tests operators of form b0^ a0^ + h.c.
    """

    active_boson_modes = [0]
    bosonic_exponents_list = [(1, 0)]
    active_fermion_modes = [0]
    fermion_operator_types = [1]
    
    operator = ParticleOperator('b0^ a0^')
    operator += operator.dagger()

    rescaling_factor = np.sqrt(maximum_occupation_number)


    be_func = partial(
        _toy_fermion_boson_interaction_be_term, active_boson_modes = active_boson_modes,
                                                bosonic_exponents_list = bosonic_exponents_list,
                                                active_fermion_modes = active_fermion_modes,
                                                fermion_operator_types = fermion_operator_types
    )

    number_of_be_ancillae = 3
    
    circuit, metrics, system = _setup(
        number_of_be_ancillae, operator, maximum_occupation_number, be_func
    )

    _validate_block_encoding(
        circuit=circuit,
        system=system,
        expected_rescaling_factor=rescaling_factor,
        operator=operator,
        number_of_block_encoding_ancillae=number_of_be_ancillae,
        maximum_occupation_number=maximum_occupation_number,
    )

    _validate_clean_ancillae_are_cleaned(
        circuit=circuit,
        system=system,
        number_of_block_encoding_ancillae=number_of_be_ancillae,
    )

    _validate_block_encoding_does_nothing_when_control_is_off(
        circuit=circuit,
        system=system,
        number_of_block_encoding_ancillae=number_of_be_ancillae,
    )

    _validate_block_encoding_select_is_self_inverse(
        circuit,
        system,
        operator,
        number_of_be_ancillae,
        maximum_occupation_number
        )

    

    

@pytest.mark.parametrize(
    "term",
    translate_antifermions_to_fermions(yukawa_hamiltonian(3, 1, 1, 1) - yukawa_hamiltonian(3, 0, 1, 1))
    .normal_order()
    .group(),
) #only test interaction terms
@pytest.mark.parametrize("reindex", [True])
def test_all_self_inverse_yukawa_terms(term, reindex):
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
    number_of_block_encoding_ancillae = 4
    maximum_occupation_number = 3
    if term == ParticleOperator('b0^ b0 a0^ a0'):
        pytest.skip()
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
        self_inverse=True
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
       
    _validate_block_encoding_select_is_self_inverse(
        circuit,
        system,
        term,
        number_of_block_encoding_ancillae,
        maximum_occupation_number
    )
