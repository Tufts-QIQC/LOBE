from openparticle import ParticleOperator
import numpy as np
from src.lobe.custom import (
    yukawa_3point_pair_term_block_encoding,
    yukawa_4point_pair_term_block_encoding,
    _custom_fermionic_plus_nonhc_block_encoding,
    _custom_term_block_encoding,
)
import pytest
from functools import partial
from _utils import (
    _setup,
    _validate_block_encoding,
    _validate_block_encoding_does_nothing_when_control_is_off,
    _validate_clean_ancillae_are_cleaned,
)


MAX_NUMBER_OF_BOSONIC_MODES = 2
MAX_NUMBER_OF_FERMIONIC_MODES = 4
POSSIBLE_MAX_OCCUPATION_NUMBERS = [1, 3, 7]


@pytest.mark.parametrize("trial", range(100))
def test_yukawa_3point(trial):
    maximum_occupation_number = int(
        np.random.choice(POSSIBLE_MAX_OCCUPATION_NUMBERS, size=1)
    )
    number_of_bosonic_modes = np.random.randint(1, MAX_NUMBER_OF_BOSONIC_MODES + 1)
    number_of_fermionic_modes = np.random.randint(2, MAX_NUMBER_OF_FERMIONIC_MODES + 1)
    bosonic_index = list(np.random.choice(range(number_of_bosonic_modes), size=1))
    fermionic_indices = list(
        np.random.choice(range(number_of_fermionic_modes), size=2, replace=False)
    )
    sign = np.random.choice([1, -1])

    operator_string = (
        f"b{fermionic_indices[1]} b{fermionic_indices[0]} a{bosonic_index[0]}^"
    )
    operator = ParticleOperator(operator_string, coeff=sign)
    conjugate_string = (
        f"b{fermionic_indices[0]}^ b{fermionic_indices[1]}^ a{bosonic_index[0]}"
    )
    operator += operator.dagger()
    # operator *= ParticleOperator("", coeff=sign)

    number_of_block_encoding_ancillae = 2
    circuit, metrics, system = _setup(
        number_of_block_encoding_ancillae,
        operator,
        maximum_occupation_number,
        partial(
            yukawa_3point_pair_term_block_encoding,
            active_indices=bosonic_index + fermionic_indices,
            sign=sign,
        ),
    )

    _validate_clean_ancillae_are_cleaned(
        circuit, system, number_of_block_encoding_ancillae
    )
    _validate_block_encoding_does_nothing_when_control_is_off(
        circuit, system, number_of_block_encoding_ancillae
    )
    _validate_block_encoding(
        circuit,
        system,
        np.sqrt(maximum_occupation_number),
        operator,
        number_of_block_encoding_ancillae,
        maximum_occupation_number,
    )

    assert (
        metrics.number_of_elbows
        == 1
        + (  # elbow for controls of adders
            2
            * (np.ceil(np.log2(maximum_occupation_number + 1)) - 1)  # elbows for adders
        )
        + np.ceil(np.log2(maximum_occupation_number + 1))  # elbows for rotation gadget
        + 1
    )  # elbow for flipping fermionic be-ancilla
    assert metrics.number_of_rotations <= (maximum_occupation_number + 3)
    assert max(metrics.clean_ancillae_usage) == max(
        2 + np.ceil(np.log2(maximum_occupation_number + 1)), 2
    )


@pytest.mark.parametrize("trial", range(100))
def test_yukawa_4point(trial):
    maximum_occupation_number = int(
        np.random.choice(POSSIBLE_MAX_OCCUPATION_NUMBERS, size=1)
    )
    number_of_bosonic_modes = np.random.randint(2, MAX_NUMBER_OF_BOSONIC_MODES + 1)
    number_of_fermionic_modes = np.random.randint(2, MAX_NUMBER_OF_FERMIONIC_MODES + 1)
    bosonic_indices = list(
        np.random.choice(range(number_of_bosonic_modes), size=2, replace=False)
    )
    fermionic_indices = list(
        np.random.choice(range(number_of_fermionic_modes), size=2, replace=False)
    )
    sign = np.random.choice([1, -1])

    operator_string = f"b{fermionic_indices[1]} b{fermionic_indices[0]} a{bosonic_indices[1]}^ a{bosonic_indices[0]}^"
    operator = ParticleOperator(operator_string, coeff=sign)
    conjugate_string = f"b{fermionic_indices[0]}^ b{fermionic_indices[1]}^ a{bosonic_indices[1]} a{bosonic_indices[0]}"
    # operator += ParticleOperator(conjugate_string)
    operator += operator.dagger()

    number_of_block_encoding_ancillae = 3
    circuit, metrics, system = _setup(
        number_of_block_encoding_ancillae,
        operator,
        maximum_occupation_number,
        partial(
            yukawa_4point_pair_term_block_encoding,
            active_indices=bosonic_indices + fermionic_indices,
            sign=sign,
        ),
    )

    _validate_clean_ancillae_are_cleaned(
        circuit, system, number_of_block_encoding_ancillae
    )
    _validate_block_encoding_does_nothing_when_control_is_off(
        circuit, system, number_of_block_encoding_ancillae
    )
    _validate_block_encoding(
        circuit,
        system,
        maximum_occupation_number,
        operator,
        number_of_block_encoding_ancillae,
        maximum_occupation_number,
    )


@pytest.mark.parametrize("trial", range(100))
def test_custom_fermionic_plus_nonhc_block_encoding(trial):
    maximum_occupation_number = 1
    expected_rescaling_factor = 1
    number_of_fermionic_modes = np.random.randint(3, MAX_NUMBER_OF_FERMIONIC_MODES + 1)
    fermionic_indices = list(
        np.random.choice(range(number_of_fermionic_modes), size=3, replace=False)
    )
    sign = np.random.choice([1, -1])

    operator_string = (
        f"b{fermionic_indices[2]} b{fermionic_indices[1]} b{fermionic_indices[0]}^"
    )
    operator = ParticleOperator(operator_string)
    nonconjugate_string = (
        f"b{fermionic_indices[1]}^ b{fermionic_indices[2]}^ b{fermionic_indices[0]}^"
    )
    operator += ParticleOperator(nonconjugate_string)
    operator *= ParticleOperator("", coeff=sign)

    number_of_block_encoding_ancillae = 1
    circuit, metrics, system = _setup(
        number_of_block_encoding_ancillae,
        operator,
        maximum_occupation_number,
        partial(
            _custom_fermionic_plus_nonhc_block_encoding,
            active_indices=fermionic_indices,
            sign=sign,
        ),
    )

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


@pytest.mark.parametrize("trial", range(100))
def test_custom_term_block_encoding(trial):
    maximum_occupation_number = np.random.choice([1, 3, 7], size=1)[0]
    expected_rescaling_factor = np.sqrt(maximum_occupation_number)
    number_of_fermionic_modes = np.random.randint(1, 1 + 1)
    active_fermionic_index = np.random.choice(range(number_of_fermionic_modes), size=1)[
        0
    ]
    number_of_bosonic_modes = np.random.randint(1, 1 + 1)
    active_bosonic_index = np.random.choice(range(number_of_bosonic_modes), size=1)[0]
    sign = np.random.choice([1, -1])

    operator_string = f"b{active_fermionic_index} a{active_bosonic_index}"
    operator = ParticleOperator(operator_string, coeff=sign)
    conjugate_string = f"b{active_fermionic_index}^ a{active_bosonic_index}^"
    # operator += ParticleOperator(conjugate_string)
    operator += operator.dagger()

    number_of_block_encoding_ancillae = 1
    circuit, metrics, system = _setup(
        number_of_block_encoding_ancillae,
        operator,
        maximum_occupation_number,
        partial(
            _custom_term_block_encoding,
            active_indices=[active_bosonic_index, active_fermionic_index],
            sign=sign,
        ),
    )

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

    assert metrics.number_of_elbows == 1 + (  # elbow for controls of adders
        2 * (np.ceil(np.log2(maximum_occupation_number + 1)) - 1)  # elbows for adders
    ) + np.ceil(
        np.log2(maximum_occupation_number + 1)
    )  # elbows for rotation gadget
    assert metrics.number_of_rotations <= (maximum_occupation_number + 3)
    assert max(metrics.clean_ancillae_usage) == max(
        1 + (len(system.bosonic_system[active_bosonic_index])), 2
    )
