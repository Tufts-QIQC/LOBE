import cirq
import pytest
import numpy as np
from functools import partial
from openparticle import ParticleOperator, generate_matrix

import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath('__file__')), '../'))
from src.lobe.self_inverse_boson import self_inverse_bosonic_number_operator_block_encoding, self_inverse_bosonic_product_plus_hc_block_encoding
from src.lobe.system import System
from src.lobe.asp import add_prepare_circuit, get_target_state
from _utils import get_basis_of_full_system

def _verify_walker_condition_one(H_matrix, basis, expected_rescaling_factor, PREPARE, SELECT, PREPARE_DAGGER):
    """
    Checks if <0|PREP^\DAGGER SEL PREP|0> = H/ALPHA
    """

    unitary = cirq.Circuit(
            PREPARE,
            SELECT,
            PREPARE_DAGGER 
        ).unitary()
    rescaled_block = (unitary * expected_rescaling_factor)[:len(basis), :len(basis)]
    assert np.allclose(
        H_matrix,
        rescaled_block
    ) 

def _verify_walker_condition_two(basis, PREPARE, SELECT, PREPARE_DAGGER):
    """
    Checks if <0|PREP^\DAGGER SEL^2 PREP|0> = 1
    """
    unitary = cirq.Circuit(
            PREPARE,
            SELECT,
            SELECT,
            PREPARE_DAGGER 
        ).unitary()
    assert np.allclose(
        np.eye(len(basis)),
        unitary[:len(basis),:len(basis)]
    )


@pytest.mark.parametrize("maximum_occupation_number", [1, 3, 7])
def test_self_inverse_bosonic_number_operator_block_encoding_on_single_mode_satisfies_walker_conditions(maximum_occupation_number):

    hamiltonian_operator = (
        ParticleOperator('a0^ a0')
    )
    rescaling_factor = maximum_occupation_number
    basis = get_basis_of_full_system(maximum_occupation_number + 1, 0, hamiltonian_operator.max_bosonic_mode + 1)
    H_matrix = generate_matrix(hamiltonian_operator, basis)

    index_register = [cirq.LineQubit(0)]
    clean_ancilla_register = [cirq.LineQubit(i) for i in range(1, 7 + 1)]
    be_anc_register = [cirq.LineQubit(8), cirq.LineQubit(9)]
    system_register = System(maximum_occupation_number, 
                        len(index_register) + len(clean_ancilla_register) + len(be_anc_register),
                        0, 
                        1)
    
    gates = []

    PREPARE = [cirq.X.on(index_register[0])]
    gates += PREPARE

    SELECT = self_inverse_bosonic_number_operator_block_encoding(
        system = system_register,
        block_encoding_ancillae=be_anc_register,
        active_mode = 0,
        sign = 1,
        clean_ancillae=clean_ancilla_register,
        ctrls = ([index_register[0]], [1])
    )

    gates += SELECT

    gates += [cirq.X.on(index_register[0])] #PREP^DAGGER


    _verify_walker_condition_one(H_matrix, basis, rescaling_factor, PREPARE, SELECT, PREPARE)
    
    _verify_walker_condition_two(basis, PREPARE, SELECT, PREPARE)

MAX_ACTIVE_MODES = 2
MAX_MODE = 1
MAX_EXPONENT = 2
@pytest.mark.parametrize("number_of_active_modes", range(1, MAX_ACTIVE_MODES + 1))
@pytest.mark.parametrize("maximum_occupation_number", [1, 3])
@pytest.mark.parametrize(
    "exponents_list",
    [
        [
            (np.random.randint(0, MAX_EXPONENT), np.random.randint(0, MAX_EXPONENT))
            for _ in range(MAX_ACTIVE_MODES)
        ]
        for _ in range(10)
    ],
)
@pytest.mark.parametrize("sign", [1.0])
def test_self_inverse_bosonic_product_plus_hc_satisfies_walker_conditions(number_of_active_modes, maximum_occupation_number, exponents_list, sign):
    active_modes = np.random.choice(
        range(MAX_MODE + 1), size=number_of_active_modes, replace=False
    )
    exponents_list = exponents_list[:number_of_active_modes]
    for i, exponents in enumerate(exponents_list):
        exponents = (
            exponents[0] % maximum_occupation_number,
            exponents[1] % maximum_occupation_number,
        )

        if exponents == (0, 0):
            exponents = (1, 0)
        exponents_list[i] = exponents

    operator_string = ""
    for i, (mode, exponents) in enumerate(
        zip(active_modes[::-1], exponents_list[::-1])
    ):
        for _ in range(exponents[0]):
            operator_string += f"a{mode}^ "
        for _ in range(exponents[1]):
            operator_string += f"a{mode} "

    operator = ParticleOperator(operator_string[:-1], coeff=1)
    operator += operator.dagger()
    print(operator)

    expected_rescaling_factor = 2
    for exponents in exponents_list:
        expected_rescaling_factor *= np.sqrt(maximum_occupation_number) ** (
            sum(exponents)
        )
    basis = get_basis_of_full_system(maximum_occupation_number, 0, operator.max_bosonic_mode + 1)
    H_matrix = generate_matrix(operator, basis)

    n_clean_ancillae = 10
    index_register = [cirq.LineQubit(0)]
    clean_ancilla_register = [cirq.LineQubit(i) for i in range(1, n_clean_ancillae + 1)]
    be_anc_register = [cirq.LineQubit(i) for i in range(n_clean_ancillae + 1, n_clean_ancillae + 1 + 2 + number_of_active_modes)]
    system = System(maximum_occupation_number, 
                    len(index_register) + len(clean_ancilla_register) + len(be_anc_register),
                    0, 
                    operator.max_bosonic_mode + 1)
    
    PREPARE = []
    for bosonic_reg in system.bosonic_modes:
        PREPARE.append(cirq.I.on_each(*bosonic_reg))
    PREPARE.append(cirq.X.on(index_register[0]))
    SELECT, _ = self_inverse_bosonic_product_plus_hc_block_encoding(
        system = system,
        block_encoding_ancillae=be_anc_register,
        active_indices=active_modes,
        exponents_list=exponents_list,
        sign = sign,
        clean_ancillae=clean_ancilla_register,
        ctrls = ([index_register[0]], [1])
    )
    PREPARE_DAGGER = [cirq.X.on(index_register[0])]

    _verify_walker_condition_one(H_matrix, basis, expected_rescaling_factor, PREPARE, SELECT, PREPARE_DAGGER)
    
    _verify_walker_condition_two(basis, PREPARE, SELECT, PREPARE_DAGGER)


# MAX_ACTIVE_MODES = 1
# MAX_MODE = 1
# MAX_EXPONENT = 1
# @pytest.mark.parametrize("number_of_active_modes", range(1, MAX_ACTIVE_MODES + 1))
# @pytest.mark.parametrize("maximum_occupation_number", [1, 3])
# @pytest.mark.parametrize(
#     "exponents_list",
#     [
#         [
#             (np.random.randint(0, MAX_EXPONENT), np.random.randint(0, MAX_EXPONENT))
#             for _ in range(MAX_ACTIVE_MODES)
#         ]
#         for _ in range(10)
#     ],
# )
# @pytest.mark.parametrize("sign", [1.0])
# def test_self_inverse_bosonic_product_plus_hc_block_encoding_satisfies_walker_conditions(
#         number_of_active_modes, 
#         maximum_occupation_number, 
#         exponents_list, 
#         sign
#     ):
#     active_modes = np.random.choice(
#         range(MAX_MODE + 1), size=number_of_active_modes, replace=False
#     )
#     exponents_list = exponents_list[:number_of_active_modes]
#     for i, exponents in enumerate(exponents_list):
#         exponents = (
#             exponents[0] % maximum_occupation_number,
#             exponents[1] % maximum_occupation_number,
#         )

#         if exponents == (0, 0):
#             exponents = (1, 0)
#         exponents_list[i] = exponents

#     operator_string = ""
#     for i, (mode, exponents) in enumerate(
#         zip(active_modes[::-1], exponents_list[::-1])
#     ):
#         for _ in range(exponents[0]):
#             operator_string += f"a{mode}^ "
#         for _ in range(exponents[1]):
#             operator_string += f"a{mode} "

#     operator = ParticleOperator(operator_string[:-1], coeff=1)
#     operator += operator.dagger()
#     print(operator)
#     basis = get_basis_of_full_system(maximum_occupation_number, 0, operator.max_bosonic_mode + 1)

#     expected_rescaling_factor = 2
#     for exponents in exponents_list:
#         expected_rescaling_factor *= np.sqrt(maximum_occupation_number) ** (
#             sum(exponents)
#         )
#     H_matrix = generate_matrix(operator, basis)

#     n_clean_ancillae = 10
#     index_register = [cirq.LineQubit(0)]
#     clean_ancilla_register = [cirq.LineQubit(i) for i in range(1, n_clean_ancillae + 1)]
#     be_anc_register = [cirq.LineQubit(i) for i in range(n_clean_ancillae + 1, n_clean_ancillae + 1 + 2 + number_of_active_modes)]
#     system = System(maximum_occupation_number, 
#                     len(index_register) + len(clean_ancilla_register) + len(be_anc_register),
#                     0, 
#                     operator.max_bosonic_mode + 1)

#     gates = []

#     PREPARE = [cirq.X.on(index_register[0])]

#     gates += PREPARE

#     SELECT = self_inverse_bosonic_product_plus_hc_block_encoding(
#         system = system,
#         block_encoding_ancillae=be_anc_register,
#         active_indices=active_modes,
#         exponents_list=exponents_list,
#         sign = 1,
#         clean_ancillae=clean_ancilla_register,
#         ctrls = ([index_register[0]], [1])
#     )[0]

#     gates += SELECT

#     PREPARE_DAGGER = [cirq.X.on(index_register[0])]
#     gates += PREPARE_DAGGER

#     _verify_walker_condition_one(H_matrix, basis, expected_rescaling_factor, PREPARE, SELECT, PREPARE_DAGGER)
    
#     _verify_walker_condition_two(basis, PREPARE, SELECT, PREPARE_DAGGER)

