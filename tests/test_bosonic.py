from openparticle import ParticleOperator, generate_matrix
from openparticle.hamiltonians.phi4_hamiltonian import phi4_Hamiltonian
import numpy as np
import cirq
from src.lobe.system import System
from src.lobe.bosonic import (
    bosonic_product_block_encoding,
    bosonic_product_plus_hc_block_encoding,
)
from src.lobe.rescale import get_number_of_active_bosonic_modes
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
from src.lobe.asp import get_target_state, add_prepare_circuit
from src.lobe.index import index_over_terms


MAX_ACTIVE_MODES = 5


@pytest.mark.parametrize("number_of_active_modes", range(1, MAX_ACTIVE_MODES + 1))
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
def test_bosonic_product_block_encoding(
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
            bosonic_product_block_encoding,
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
    assert metrics.number_of_nonclifford_rotations <= number_of_active_modes * (
        maximum_occupation_number + 3
    )
    assert len(metrics.rotation_angles) == number_of_active_modes * (
        maximum_occupation_number + 3
    )
    assert metrics.clean_ancillae_usage[-1] == 0
    assert metrics.ancillae_highwater() == np.ceil(
        np.log2(maximum_occupation_number + 1)
    )


@pytest.mark.parametrize("number_of_active_modes", range(1, MAX_ACTIVE_MODES + 1))
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
    assert metrics.number_of_nonclifford_rotations <= number_of_active_modes * (
        maximum_occupation_number + 3
    )
    assert len(metrics.rotation_angles) == number_of_active_modes * (
        maximum_occupation_number + 3
    )
    assert metrics.clean_ancillae_usage[-1] == 0
    assert (
        metrics.ancillae_highwater()
        == np.ceil(np.log2(maximum_occupation_number + 1)) + 1
    )


@pytest.mark.parametrize(
    "term",
    [
        ParticleOperator("a0^ a0^ a0 a0"),
        ParticleOperator("a1^ a1^ a1 a1"),
        ParticleOperator("a1^ a0^ a1 a0"),
        ParticleOperator("a1^ a1"),
        ParticleOperator("a0^ a0"),
    ],
)
def test_phi4_term(term):
    operator = term
    maximum_occupation_number = 3
    active_modes, exponents_list = get_bosonic_exponents(
        operator, operator.max_bosonic_mode + 1
    )
    number_of_active_modes = len(active_modes)
    sign = 1

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
            bosonic_product_block_encoding,
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
    assert metrics.number_of_nonclifford_rotations <= number_of_active_modes * (
        maximum_occupation_number + 3
    )
    assert len(metrics.rotation_angles) == number_of_active_modes * (
        maximum_occupation_number + 3
    )
    assert metrics.clean_ancillae_usage[-1] == 0


@pytest.mark.parametrize(
    "operator",
    [
        ParticleOperator("a1^ a1^ a1 a1") + ParticleOperator("a0^ a0"),
        ParticleOperator("a0^ a0^ a0 a0") + ParticleOperator("a1^ a1"),
        ParticleOperator("a1^ a1^ a1 a1") + ParticleOperator("a1^ a1"),
        ParticleOperator("a0^ a0^ a0 a0") + ParticleOperator("a0^ a0"),
        ParticleOperator("a0^ a0") + ParticleOperator("a1^ a1"),
        ParticleOperator("a1^ a0^ a1 a0") + ParticleOperator("a1^ a1^ a1 a1"),
        ParticleOperator("a1^ a0^ a1 a0") + ParticleOperator("a0^ a0^ a0 a0"),
    ],
)
def test_phi4_sum_of_self_conjugate_terms(operator):
    maximum_occupation_number = 3

    grouped_terms = operator.to_list()
    number_of_block_encoding_ancillae = max(
        get_number_of_active_bosonic_modes(grouped_terms)
    )

    index_register = [
        cirq.LineQubit(-i - 2) for i in range(int(np.ceil(np.log2(len(grouped_terms)))))
    ]
    block_encoding_ancillae = [
        cirq.LineQubit(-100 - i - len(index_register))
        for i in range(number_of_block_encoding_ancillae)
    ]
    ctrls = ([cirq.LineQubit(0)], [1])
    clean_ancillae = [cirq.LineQubit(i + 100) for i in range(100)]
    system = System(
        operator.max_bosonic_mode + 1,
        maximum_occupation_number,
        1000,
        False,
        False,
        True,
    )

    BE_functions = []
    rescaling_factors = []
    for term in grouped_terms:
        active_modes, exponents_list = get_bosonic_exponents(
            term, term.max_bosonic_mode + 1
        )

        BE_functions.append(
            partial(
                bosonic_product_block_encoding,
                system=system,
                block_encoding_ancillae=block_encoding_ancillae,
                active_indices=active_modes,
                exponents_list=exponents_list,
                clean_ancillae=clean_ancillae[1:],
            )
        )
        rescaling_factors.append(
            np.sqrt(maximum_occupation_number)
            ** (sum([sum(exponents) for exponents in exponents_list]))
        )

    rescaled_coefficients = []
    for term, rescaling_factor in zip(grouped_terms, rescaling_factors):
        rescaled_coefficients.append(
            term.coeffs[0] * rescaling_factor / max(rescaling_factors)
        )

    target_state = get_target_state(rescaled_coefficients)
    gates = []
    for mode in system.bosonic_system:
        for qubit in mode:
            gates.append(cirq.I.on(qubit))

    gates.append(cirq.X.on(ctrls[0][0]))

    _gates, _ = add_prepare_circuit(
        index_register, target_state, clean_ancillae=clean_ancillae
    )
    gates += _gates

    _gates, _ = index_over_terms(
        index_register, BE_functions, clean_ancillae, ctrls=ctrls
    )
    gates += _gates

    _gates, _ = add_prepare_circuit(
        index_register, target_state, dagger=True, clean_ancillae=clean_ancillae
    )
    gates += _gates

    gates.append(cirq.X.on(ctrls[0][0]))

    overall_rescaling_factor = sum(
        [
            term.coeffs[0] * rescaling_factor
            for term, rescaling_factor in zip(grouped_terms, rescaling_factors)
        ]
    )

    unitary = (
        cirq.Circuit(gates).unitary()[
            : 1 << system.number_of_system_qubits, : 1 << system.number_of_system_qubits
        ]
        * overall_rescaling_factor
    )

    full_fock_basis = get_basis_of_full_system(
        operator.max_bosonic_mode + 1, maximum_occupation_number, has_bosons=True
    )
    matrix = generate_matrix(operator, full_fock_basis)

    if not np.allclose(unitary, matrix):
        print("unitary\n", unitary.real.round(1))
        print("expected\n", matrix.real.round(1))
        assert False
    assert np.allclose(unitary, matrix)


@pytest.mark.parametrize("res", np.arange(2, 4, 1))
@pytest.mark.parametrize("maximum_occupation_number", [1, 3])
def test_phi4_hamiltonian_block_encoding(res, maximum_occupation_number):
    operator = phi4_Hamiltonian(res, 1, 1)

    grouped_terms = operator.group()
    number_of_block_encoding_ancillae = max(
        get_number_of_active_bosonic_modes(grouped_terms)
    )

    index_register = [
        cirq.LineQubit(-i - 2) for i in range(int(np.ceil(np.log2(len(grouped_terms)))))
    ]
    block_encoding_ancillae = [
        cirq.LineQubit(-100 - i - len(index_register))
        for i in range(number_of_block_encoding_ancillae)
    ]
    ctrls = ([cirq.LineQubit(0)], [1])
    clean_ancillae = [cirq.LineQubit(i + 100) for i in range(100)]
    system = System(
        operator.max_bosonic_mode + 1,
        maximum_occupation_number,
        1000,
        False,
        False,
        True,
    )

    block_encoding_functions = []
    rescaling_factors = []
    for term in grouped_terms:
        plus_hc = False
        if len(term) == 2:
            plus_hc = True
            term = term.to_list()[0]
        active_modes, exponents = get_bosonic_exponents(
            term, operator.max_bosonic_mode + 1
        )

    #         if not plus_hc:
    #             block_encoding_functions.append(
    #                 partial(
    #                     bosonic_product_block_encoding,
    #                     system=system,
    #                     block_encoding_ancillae=block_encoding_ancillae,
    #                     active_indices=active_modes,
    #                     exponents_list=exponents,
    #                     clean_ancillae=clean_ancillae[1:],
    #                 )
    #             )
    #             rescaling_factors.append(
    #                 np.sqrt(maximum_occupation_number) ** (sum(sum(np.asarray(exponents))))
    #             )
    #         else:
    #             block_encoding_functions.append(
    #                 partial(
    #                     bosonic_product_plus_hc_block_encoding,
    #                     system=system,
    #                     block_encoding_ancillae=block_encoding_ancillae,
    #                     active_indices=active_modes,
    #                     exponents_list=exponents,
    #                     clean_ancillae=clean_ancillae[1:],
    #                 )
    #             )
    #             rescaling_factors.append(
    #                 2
    #                 * np.sqrt(maximum_occupation_number)
    #                 ** (sum(sum(np.asarray(exponents))))
    #             )

    rescaled_coefficients = []
    for term, rescaling_factor in zip(grouped_terms, rescaling_factors):
        rescaled_coefficients.append(
            term.coeffs[0] * rescaling_factor / max(rescaling_factors)
        )

    target_state = get_target_state(rescaled_coefficients)
    gates = []
    for mode in system.bosonic_system:
        for qubit in mode:
            gates.append(cirq.I.on(qubit))

    gates.append(cirq.X.on(ctrls[0][0]))

    _gates, _ = add_prepare_circuit(
        index_register, target_state, clean_ancillae=clean_ancillae
    )
    gates += _gates

    _gates, _ = index_over_terms(
        index_register, block_encoding_functions, clean_ancillae, ctrls=ctrls
    )
    gates += _gates

    _gates, _ = add_prepare_circuit(
        index_register, target_state, dagger=True, clean_ancillae=clean_ancillae
    )
    gates += _gates

    gates.append(cirq.X.on(ctrls[0][0]))

    overall_rescaling_factor = sum(
        [
            term.coeffs[0] * rescaling_factor
            for term, rescaling_factor in zip(grouped_terms, rescaling_factors)
        ]
    )

    unitary = (
        cirq.Circuit(gates).unitary()[
            : 1 << system.number_of_system_qubits, : 1 << system.number_of_system_qubits
        ]
        * overall_rescaling_factor
    )

    full_fock_basis = get_basis_of_full_system(
        operator.max_bosonic_mode + 1, maximum_occupation_number, has_bosons=True
    )
    matrix = generate_matrix(operator, full_fock_basis)

    if not np.allclose(unitary, matrix):
        print("unitary\n", unitary.real.round(1))
        print("expected\n", matrix.real.round(1))
        assert False
    assert True
