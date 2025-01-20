from openparticle import ParticleOperator, generate_matrix
import numpy as np
import cirq
from src.lobe.system import System
from src.lobe.bosonic import (
    bosonic_mode_block_encoding,
    bosonic_modes_block_encoding,
    bosonic_mode_plus_hc_block_encoding,
    bosonic_product_plus_hc_block_encoding,
)
import pytest
from functools import partial

from _utils import (
    _setup,
    _validate_block_encoding,
    _validate_clean_ancillae_are_cleaned,
    _validate_block_encoding_does_nothing_when_control_is_off,
    get_basis_of_full_system,
)
from src.lobe._utils import get_bosonic_exponents

from src.lobe.rescale import bosonically_rescale_terms
from src.lobe.asp import get_target_state


@pytest.mark.parametrize("number_of_modes", range(2, 4))
@pytest.mark.parametrize("active_mode", range(0, 4))
@pytest.mark.parametrize("maximum_occupation_number", [1, 3, 7])
@pytest.mark.parametrize("R", range(1, 4))
@pytest.mark.parametrize("S", range(1, 4))
@pytest.mark.parametrize("sign", [1, -1])
def test_bosonic_mode_block_encoding(
    number_of_modes, active_mode, maximum_occupation_number, R, S, sign
):
    if (maximum_occupation_number == 7) and (active_mode > 0):
        pytest.skip()

    active_mode = active_mode % number_of_modes
    operator = ParticleOperator(f"a{active_mode}^") ** R
    operator *= ParticleOperator(f"a{active_mode}") ** S
    operator *= ParticleOperator("", coeff=sign)
    expected_rescaling_factor = np.sqrt(maximum_occupation_number) ** (R + S)

    number_of_block_encoding_ancillae = 1
    circuit, metrics, system = _setup(
        number_of_block_encoding_ancillae,
        operator,
        maximum_occupation_number,
        partial(
            bosonic_mode_block_encoding,
            active_index=active_mode,
            exponents=(R, S),
            sign=sign,
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
    assert metrics.number_of_elbows <= np.ceil(
        np.log2(maximum_occupation_number + 1)
    ) + max(int(np.log2(maximum_occupation_number + 1)) - 1, 0)
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
@pytest.mark.parametrize("sign", [1, -1])
def test_bosonic_modes_block_encoding(
    number_of_active_modes, maximum_occupation_number, exponents_list, sign
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
    operator *= ParticleOperator("", coeff=sign)
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
            sign=sign,
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
        np.ceil(np.log2(maximum_occupation_number + 1))
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
@pytest.mark.parametrize("sign", [1, -1])
def test_bosonic_mode_plus_hc_block_encoding(
    number_of_modes, active_mode, maximum_occupation_number, R, S, sign
):
    if R == S:
        pytest.skip()
    if (maximum_occupation_number == 7) and (active_mode > 0):
        pytest.skip()

    active_mode = active_mode % number_of_modes
    operator = ParticleOperator(f"a{active_mode}^") ** R
    operator *= ParticleOperator(f"a{active_mode}") ** S
    operator *= ParticleOperator("", coeff=sign)
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
            sign=sign,
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
        3 * int(np.log2(maximum_occupation_number + 1)), 0
    )
    assert metrics.number_of_rotations <= maximum_occupation_number + 2
    assert metrics.clean_ancillae_usage[-1] == 0


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
@pytest.mark.parametrize("sign", [1, -1])
def test_bosonic_product_plus_hc_block_encoding(
    number_of_active_modes, maximum_occupation_number, exponents_list, sign
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

    operator = ParticleOperator(operator_string[:-1], coeff=sign)
    operator += operator.dagger()

    expected_rescaling_factor = 2
    for exponents in exponents_list:
        expected_rescaling_factor *= np.sqrt(maximum_occupation_number) ** (
            sum(exponents)
        )

    number_of_block_encoding_ancillae = number_of_active_modes + 1
    circuit, metrics, system = _setup(
        number_of_block_encoding_ancillae,
        operator,
        maximum_occupation_number,
        partial(
            bosonic_product_plus_hc_block_encoding,
            active_indices=active_modes,
            exponents_list=exponents_list,
            sign=sign,
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
    assert (
        metrics.number_of_elbows
        <= (number_of_active_modes)
        * (
            np.ceil(np.log2(maximum_occupation_number + 1))  # rotations
            + (2 * max(int(np.log2(maximum_occupation_number + 1)) - 1, 0))  # adder
        )
        + 1  # toggle between terms
    )
    assert metrics.number_of_rotations <= number_of_active_modes * (
        maximum_occupation_number + 3
    )
    assert metrics.clean_ancillae_usage[-1] == 0


@pytest.mark.parametrize(
    "term",
    [
        ParticleOperator("a1^ a1^ a1 a1"),
        ParticleOperator("a1^ a0^ a0 a1"),
        ParticleOperator("a1^ a1"),
        ParticleOperator("a0^ a0^ a0 a0"),
        ParticleOperator("a0^ a0"),
    ],
)
def test_phi4_individual_term(term):
    max_bose_occ = 3
    active_modes, exponents = get_bosonic_exponents(term, term.max_bosonic_mode + 1)

    terms = term.to_list()

    bosonically_rescaled_terms, bosonic_rescaling_factor = bosonically_rescale_terms(
        terms, max_bose_occ
    )
    coefficients = [term.coeff for term in bosonically_rescaled_terms]

    norm = sum(np.abs(coefficients))
    target_state = get_target_state(coefficients)
    asp_rescaling_factor = bosonic_rescaling_factor * norm

    number_of_modes = max([term.max_mode for term in terms]) + 1

    number_of_ancillae = (
        1000  # Some arbitrary large number with most ancilla disregarded
    )
    number_of_index_qubits = max(int(np.ceil(np.log2(len(terms)))), 1)
    number_of_rotation_qubits = len(active_modes)

    # Declare Qubits
    control = cirq.LineQubit(0)
    clean_ancillae = [cirq.LineQubit(i + 1) for i in range(number_of_ancillae)]

    block_encoding_ancillae = [
        cirq.LineQubit(number_of_index_qubits + i + number_of_ancillae)
        for i in range(number_of_rotation_qubits + 1)
    ]  # Index register + rotation qubits

    system = System(
        number_of_modes=number_of_modes,
        maximum_occupation_number=max_bose_occ,
        number_of_used_qubits=1
        + number_of_ancillae
        + number_of_rotation_qubits
        + number_of_index_qubits,
        has_fermions=term.has_fermions,
        has_antifermions=term.has_antifermions,
        has_bosons=term.has_bosons,
    )
    gates = []
    gates.append(cirq.X.on(control))
    _gates, metrics = bosonic_modes_block_encoding(
        system,
        block_encoding_ancillae,
        active_modes,
        exponents,
        clean_ancillae=clean_ancillae,
        ctrls=([control], [1]),
    )
    gates += _gates
    gates.append(cirq.X.on(control))

    unitary_rescaling_factor = 2 * np.sqrt(max_bose_occ) ** sum(
        sum(np.asarray(exponents))
    )

    unitary = (
        cirq.Circuit(gates).unitary()[
            : 1 << system.number_of_system_qubits, : 1 << system.number_of_system_qubits
        ]
        * unitary_rescaling_factor
    )
    basis = get_basis_of_full_system(2, max_bose_occ, False, False, True)
    matrix = generate_matrix(term + term.dagger(), basis)


@pytest.mark.parametrize(
    "term",
    [
        ParticleOperator("a1^ a1^ a1 a1"),
        ParticleOperator("a1^ a0^ a0 a1"),
        ParticleOperator("a1^ a1"),
        ParticleOperator("a0^ a0^ a0 a0"),
        ParticleOperator("a0^ a0"),
    ],
)
def test_phi4_individual_term_plus_hc(term):
    max_bose_occ = 3
    active_modes, exponents = get_bosonic_exponents(term, term.max_bosonic_mode + 1)

    terms = term.to_list()

    bosonically_rescaled_terms, bosonic_rescaling_factor = bosonically_rescale_terms(
        terms, max_bose_occ
    )
    coefficients = [term.coeff for term in bosonically_rescaled_terms]

    norm = sum(np.abs(coefficients))
    target_state = get_target_state(coefficients)
    asp_rescaling_factor = bosonic_rescaling_factor * norm

    number_of_modes = max([term.max_mode for term in terms]) + 1

    number_of_ancillae = (
        1000  # Some arbitrary large number with most ancilla disregarded
    )
    number_of_index_qubits = max(int(np.ceil(np.log2(len(terms)))), 1)
    number_of_rotation_qubits = len(active_modes)

    # Declare Qubits
    control = cirq.LineQubit(0)
    clean_ancillae = [cirq.LineQubit(i + 1) for i in range(number_of_ancillae)]

    block_encoding_ancillae = [
        cirq.LineQubit(number_of_index_qubits + i + number_of_ancillae)
        for i in range(number_of_rotation_qubits + 1)
    ]  # Index register + rotation qubits

    system = System(
        number_of_modes=number_of_modes,
        maximum_occupation_number=max_bose_occ,
        number_of_used_qubits=1
        + number_of_ancillae
        + number_of_rotation_qubits
        + number_of_index_qubits,
        has_fermions=term.has_fermions,
        has_antifermions=term.has_antifermions,
        has_bosons=term.has_bosons,
    )
    gates = []
    gates.append(cirq.X.on(control))
    _gates, metrics = bosonic_product_plus_hc_block_encoding(
        system,
        block_encoding_ancillae,
        active_modes,
        exponents,
        clean_ancillae=clean_ancillae,
        ctrls=([control], [1]),
    )
    gates += _gates
    gates.append(cirq.X.on(control))

    unitary_rescaling_factor = 2 * np.sqrt(max_bose_occ) ** sum(
        sum(np.asarray(exponents))
    )

    unitary = (
        cirq.Circuit(gates).unitary()[
            : 1 << system.number_of_system_qubits, : 1 << system.number_of_system_qubits
        ]
        * unitary_rescaling_factor
    )
    basis = get_basis_of_full_system(2, max_bose_occ, False, False, True)
    matrix = generate_matrix(term + term.dagger(), basis)
