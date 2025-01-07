from openparticle import ParticleOperator
import numpy as np
from src.lobe.bosonic import (
    bosonic_mode_block_encoding,
    bosonic_modes_block_encoding,
    bosonic_mode_plus_hc_block_encoding,
)
import pytest
from functools import partial
from _utils import (
    _setup,
    _validate_block_encoding,
    _validate_clean_ancillae_are_cleaned,
    _validate_block_encoding_does_nothing_when_control_is_off,
)


@pytest.mark.parametrize("number_of_modes", range(2, 4))
@pytest.mark.parametrize("active_mode", range(0, 4))
@pytest.mark.parametrize("maximum_occupation_number", [1, 3, 7])
@pytest.mark.parametrize("R", range(1, 4))
@pytest.mark.parametrize("S", range(1, 4))
def test_bosonic_mode_block_encoding(
    number_of_modes, active_mode, maximum_occupation_number, R, S
):
    if (maximum_occupation_number == 7) and (active_mode > 0):
        pytest.skip()

    active_mode = active_mode % number_of_modes
    operator = ParticleOperator(f"a{active_mode}^") ** R
    operator *= ParticleOperator(f"a{active_mode}") ** S
    expected_rescaling_factor = np.sqrt(maximum_occupation_number) ** (R + S)

    number_of_block_encoding_ancillae = 1
    circuit, metrics, system = _setup(
        number_of_block_encoding_ancillae,
        operator,
        maximum_occupation_number,
        partial(
            bosonic_mode_block_encoding, active_index=active_mode, exponents=(R, S)
        ),
    )
    _validate_block_encoding(
        circuit,
        system,
        expected_rescaling_factor,
        operator,
        number_of_block_encoding_ancillae,
        maximum_occupation_number,
    )
    _validate_clean_ancillae_are_cleaned(
        circuit,
        system,
        number_of_block_encoding_ancillae,
    )
    _validate_block_encoding_does_nothing_when_control_is_off(
        circuit, system, number_of_block_encoding_ancillae
    )
    assert metrics.number_of_elbows <= maximum_occupation_number + 1 + max(
        int(np.log2(maximum_occupation_number + 1)) - 1, 0
    )
    assert metrics.number_of_rotations <= maximum_occupation_number + 2
    assert metrics.clean_ancillae_usage[-1] == 0


MAX_ACTIVE_MODES = 5


@pytest.mark.parametrize("number_of_active_modes", range(2, MAX_ACTIVE_MODES + 1))
@pytest.mark.parametrize("maximum_occupation_number", [1, 3, 7])
@pytest.mark.parametrize(
    "exponents_list",
    [
        [
            (np.random.randint(0, 4), np.random.randint(0, 4))
            for _ in range(MAX_ACTIVE_MODES)
        ]
        for _ in range(10)
    ],
)
def test_bosonic_modes_block_encoding(
    number_of_active_modes, maximum_occupation_number, exponents_list
):
    active_modes = np.random.choice(
        range(MAX_ACTIVE_MODES), size=number_of_active_modes, replace=False
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

    operator = ParticleOperator(operator_string[:-1])
    expected_rescaling_factor = 1
    for exponents in exponents_list:
        expected_rescaling_factor *= np.sqrt(maximum_occupation_number) ** (
            sum(exponents)
        )

    number_of_block_encoding_ancillae = number_of_active_modes
    circuit, metrics, system = _setup(
        number_of_block_encoding_ancillae,
        operator,
        maximum_occupation_number,
        partial(
            bosonic_modes_block_encoding,
            active_indices=active_modes,
            exponents_list=exponents_list,
        ),
    )
    _validate_block_encoding(
        circuit,
        system,
        expected_rescaling_factor,
        operator,
        number_of_block_encoding_ancillae,
        maximum_occupation_number,
    )
    _validate_clean_ancillae_are_cleaned(
        circuit,
        system,
        number_of_block_encoding_ancillae,
    )
    _validate_block_encoding_does_nothing_when_control_is_off(
        circuit, system, number_of_block_encoding_ancillae
    )
    assert metrics.number_of_elbows <= (number_of_active_modes) * (
        maximum_occupation_number
        + 1
        + max(int(np.log2(maximum_occupation_number + 1)) - 1, 0)
    )
    assert metrics.number_of_rotations <= number_of_active_modes * (
        maximum_occupation_number + 3
    )
    assert metrics.clean_ancillae_usage[-1] == 0


@pytest.mark.parametrize("number_of_modes", range(2, 4))
@pytest.mark.parametrize("active_mode", range(0, 4))
@pytest.mark.parametrize("maximum_occupation_number", [1, 3, 7])
@pytest.mark.parametrize("R", range(1, 4))
@pytest.mark.parametrize("S", range(1, 4))
def test_bosonic_mode_plus_hc_block_encoding(
    number_of_modes, active_mode, maximum_occupation_number, R, S
):
    if R == S:
        pytest.skip()
    if (maximum_occupation_number == 7) and (active_mode > 0):
        pytest.skip()

    active_mode = active_mode % number_of_modes
    operator = ParticleOperator(f"a{active_mode}^") ** R
    operator *= ParticleOperator(f"a{active_mode}") ** S
    operator += operator.dagger()
    expected_rescaling_factor = 2 * np.sqrt(maximum_occupation_number) ** (R + S)

    number_of_block_encoding_ancillae = 2
    circuit, metrics, system = _setup(
        number_of_block_encoding_ancillae,
        operator,
        maximum_occupation_number,
        partial(
            bosonic_mode_plus_hc_block_encoding,
            active_index=active_mode,
            exponents=(R, S),
        ),
    )
    _validate_block_encoding(
        circuit,
        system,
        expected_rescaling_factor,
        operator,
        number_of_block_encoding_ancillae,
        maximum_occupation_number,
    )
    _validate_clean_ancillae_are_cleaned(
        circuit,
        system,
        number_of_block_encoding_ancillae,
    )
    _validate_block_encoding_does_nothing_when_control_is_off(
        circuit, system, number_of_block_encoding_ancillae
    )
    assert metrics.number_of_elbows <= max(
        maximum_occupation_number + (2 * int(np.log2(maximum_occupation_number + 1))), 0
    )
    assert metrics.number_of_rotations <= maximum_occupation_number + 2
    assert metrics.clean_ancillae_usage[-1] == 0
