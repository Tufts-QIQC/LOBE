import pytest
import numpy as np
from functools import partial
from openparticle import ParticleOperator
from src.lobe.fermionic import (
    fermionic_plus_hc_block_encoding,
    fermionic_product_block_encoding,
)
from _utils import (
    _setup,
    _validate_block_encoding,
    _validate_clean_ancillae_are_cleaned,
    _validate_block_encoding_does_nothing_when_control_is_off,
)


MAX_MODES = 7
MAX_ACTIVE_MODES = 7
MIN_ACTIVE_MODES = 1


@pytest.mark.parametrize("trial", range(100))
def test_arbitrary_fermionic_operator_with_hc(trial):
    number_of_active_modes = np.random.randint(MIN_ACTIVE_MODES, MAX_ACTIVE_MODES + 1)
    active_modes = np.random.choice(
        range(MAX_MODES + 1), size=number_of_active_modes, replace=False
    )
    operator_types_reversed = np.random.choice(
        [2, 1, 0], size=number_of_active_modes, replace=True
    )
    while np.allclose(operator_types_reversed, [2] * number_of_active_modes):
        operator_types_reversed = np.random.choice(
            [2, 1, 0], size=number_of_active_modes, replace=True
        )
    operator_types_reversed = operator_types_reversed[:number_of_active_modes]
    operator_types_reversed = list(operator_types_reversed)
    sign = np.random.choice([1, -1])

    operator_string = ""
    for mode, operator_type in zip(active_modes, operator_types_reversed):
        if operator_type == 0:
            operator_string += f" b{mode}"
        if operator_type == 1:
            operator_string += f" b{mode}^"
        if operator_type == 2:
            operator_string += f" b{mode}^ b{mode}"

    operator = ParticleOperator(operator_string, coeff=sign)
    operator += operator.dagger()

    circuit, metrics, system = _setup(
        1,
        operator,
        1,
        partial(
            fermionic_plus_hc_block_encoding,
            active_indices=active_modes[::-1],
            operator_types=operator_types_reversed[::-1],
            sign=sign,
        ),
    )

    number_of_block_encoding_ancillae = 1
    expected_rescaling_factor = 1
    maximum_occupation_number = 1
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
        operator,
        number_of_block_encoding_ancillae,
        maximum_occupation_number,
    )

    number_of_number_ops = operator_types_reversed[::-1].count(2)

    assert metrics.number_of_elbows == len(active_modes[::-1]) - 1
    if len(metrics.clean_ancillae_usage) > 0:
        assert metrics.clean_ancillae_usage[-1] == 0
        assert (
            max(metrics.clean_ancillae_usage)
            == (len(active_modes[::-1]) - 2)  # elbows for qbool
            + 1  # elbow to apply toff that flips ancilla
        )
    else:
        assert number_of_active_modes == 1


@pytest.mark.parametrize("trial", range(100))
def test_arbitrary_fermionic_product(trial):
    number_of_active_modes = np.random.randint(MIN_ACTIVE_MODES, MAX_ACTIVE_MODES + 1)
    active_modes = np.random.choice(
        range(MAX_MODES + 1), size=number_of_active_modes, replace=False
    )
    operator_types_reversed = np.random.choice(
        [2, 1, 0], size=number_of_active_modes, replace=True
    )
    while np.allclose(operator_types_reversed, [2] * number_of_active_modes):
        operator_types_reversed = np.random.choice(
            [2, 1, 0], size=number_of_active_modes, replace=True
        )
    operator_types_reversed = operator_types_reversed[:number_of_active_modes]
    operator_types_reversed = list(operator_types_reversed)
    sign = np.random.choice([1, -1])

    operator_string = ""
    for mode, operator_type in zip(active_modes, operator_types_reversed):
        if operator_type == 0:
            operator_string += f" b{mode}"
        if operator_type == 1:
            operator_string += f" b{mode}^"
        if operator_type == 2:
            operator_string += f" b{mode}^ b{mode}"

    operator = sign * ParticleOperator(operator_string)

    circuit, metrics, system = _setup(
        1,
        operator,
        1,
        partial(
            fermionic_product_block_encoding,
            active_indices=active_modes[::-1],
            operator_types=operator_types_reversed[::-1],
            sign=sign,
        ),
    )

    number_of_block_encoding_ancillae = 1
    expected_rescaling_factor = 1
    maximum_occupation_number = 1
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
        operator,
        number_of_block_encoding_ancillae,
        maximum_occupation_number,
    )

    assert metrics.number_of_elbows == len(active_modes[::-1])
    assert metrics.clean_ancillae_usage[-1] == 0
    assert (
        max(metrics.clean_ancillae_usage)
        == (len(active_modes[::-1]) - 1)  # elbows for qbool
        + 1  # elbow to apply toff that flips ancilla
    )
