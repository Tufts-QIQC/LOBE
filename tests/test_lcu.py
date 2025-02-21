import pytest
from functools import partial
from openparticle import ParticleOperator
from src.lobe.lcu import (
    pauli_lcu_block_encoding,
    estimate_pauli_lcu_rescaling_factor_and_number_of_be_ancillae,
)
from src.lobe.system import System
from _utils import (
    _setup,
    _validate_block_encoding,
    _validate_block_encoding_does_nothing_when_control_is_off,
    _validate_clean_ancillae_are_cleaned,
)


@pytest.mark.parametrize(
    "op, max_bosonic_occupancy",
    [
        [ParticleOperator("b0"), 1],
        [ParticleOperator("a0 a0"), 3],
        [ParticleOperator("a0 a0"), 7],
        [ParticleOperator("a0"), 1],
        [ParticleOperator("a0"), 3],
        [ParticleOperator("a0"), 7],
        [ParticleOperator("a1"), 1],
        [ParticleOperator("a1"), 3],
        [ParticleOperator("a1"), 7],
        [ParticleOperator("a1"), 15],
        [ParticleOperator("a1"), 15],
        [ParticleOperator("a0 a1"), 3],
        [ParticleOperator("a0") + ParticleOperator("a2"), 3],
        [ParticleOperator("a0 a0") + ParticleOperator("a0^ a0^"), 3],
        [ParticleOperator("b0 b1"), 1],
        [ParticleOperator("b0 b1"), 1],
        [ParticleOperator("b1 b2"), 1],
        [ParticleOperator("b0 b1 b2"), 1],
        [ParticleOperator("b1^ b0 b2"), 1],
    ],
)
def test_functional_lcu(op, max_bosonic_occupancy):
    number_of_fermionic_modes = 0
    number_of_bosonic_modes = 0
    if op.max_fermionic_mode is not None:
        number_of_fermionic_modes = op.max_fermionic_mode + 1
    if op.max_bosonic_mode is not None:
        number_of_bosonic_modes = op.max_bosonic_mode + 1
    system = System(
        max_bosonic_occupancy,
        0,
        number_of_fermionic_modes=number_of_fermionic_modes,
        number_of_bosonic_modes=number_of_bosonic_modes,
    )

    expected_rescaling_factor, number_of_block_encoding_ancillae = (
        estimate_pauli_lcu_rescaling_factor_and_number_of_be_ancillae(
            system, op, zero_threshold=1e-6
        )
    )
    paulis = op.to_paulis(
        max_fermionic_mode=op.max_fermionic_mode,
        max_antifermionic_mode=op.max_antifermionic_mode,
        max_bosonic_mode=op.max_bosonic_mode,
        max_bosonic_occupancy=system.maximum_occupation_number,
        zero_threshold=1e-12,
    )
    system = System(
        max_bosonic_occupancy,
        1 + number_of_block_encoding_ancillae + 100,
        number_of_fermionic_modes=number_of_fermionic_modes,
        number_of_bosonic_modes=number_of_bosonic_modes,
    )

    system_register = system.fermionic_modes[::-1]
    for bosonic_reg in system.bosonic_modes[::-1]:
        system_register += bosonic_reg

    block_encoding_function = partial(
        pauli_lcu_block_encoding,
        system_register=system_register,
        paulis=paulis,
        zero_threshold=1e-6,
    )
    circuit, metrics, system = _setup(
        number_of_block_encoding_ancillae,
        op,
        max_bosonic_occupancy,
        block_encoding_function,
    )
    _validate_clean_ancillae_are_cleaned(
        circuit,
        system,
        number_of_block_encoding_ancillae,
    )
    _validate_block_encoding_does_nothing_when_control_is_off(
        circuit,
        system,
        number_of_block_encoding_ancillae,
    )
    _validate_block_encoding(
        circuit,
        system,
        expected_rescaling_factor,
        op,
        number_of_block_encoding_ancillae,
        max_bosonic_occupancy,
    )
