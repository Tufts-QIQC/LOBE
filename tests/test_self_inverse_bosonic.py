import cirq
import pytest
import numpy as np
from functools import partial
from openparticle import ParticleOperator, generate_matrix

import sys, os

sys.path.append(os.path.join(os.path.dirname(os.path.realpath("__file__")), "../"))
from src.lobe.bosonic import (
    self_inverse_bosonic_number_operator_block_encoding,
    self_inverse_bosonic_product_plus_hc_block_encoding,
)
from src.lobe.system import System
from src.lobe.asp import add_prepare_circuit, get_target_state
from _utils import get_basis_of_full_system
from tests._utils import (
    _setup,
    _validate_block_encoding,
    _validate_block_encoding_does_nothing_when_control_is_off,
    _validate_block_encoding_select_is_self_inverse,
    _validate_clean_ancillae_are_cleaned,
)


@pytest.mark.parametrize("maximum_occupation_number", [1, 3, 7])
def test_self_inverse_bosonic_number_operator_block_encoding_on_single_mode_satisfies_walker_conditions(
    maximum_occupation_number,
):

    operator = ParticleOperator("a0^ a0")
    rescaling_factor = maximum_occupation_number

    be_func = partial(
        self_inverse_bosonic_number_operator_block_encoding, active_mode=0, sign=1
    )
    number_of_be_ancillae = 2
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

    _validate_block_encoding_select_is_self_inverse(circuit)


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
def test_self_inverse_bosonic_product_plus_hc_satisfies_walker_conditions(
    number_of_active_modes, maximum_occupation_number, exponents_list, sign
):
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

    expected_rescaling_factor = 2
    for exponents in exponents_list:
        expected_rescaling_factor *= np.sqrt(maximum_occupation_number) ** (
            sum(exponents)
        )

    be_func = partial(
        self_inverse_bosonic_product_plus_hc_block_encoding,
        active_indices=active_modes,
        exponents_list=exponents_list,
        sign=sign,
    )

    number_of_be_ancillae = 2 + number_of_active_modes
    circuit, metrics, system = _setup(
        number_of_be_ancillae, operator, maximum_occupation_number, be_func
    )

    _validate_block_encoding(
        circuit=circuit,
        system=system,
        expected_rescaling_factor=expected_rescaling_factor,
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

    _validate_block_encoding_select_is_self_inverse(circuit)
