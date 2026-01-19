import cirq
import numpy as np
from openparticle import ParticleOperator

from .addition import add_classical_value
from .bosonic import _get_bosonic_rotation_angles
from .metrics import CircuitMetrics
from .multiplexed_rotations import _process_rotation_angles
from ._utils import (
    get_bosonic_exponents,
    translate_antifermions_to_fermions,
    get_active_bosonic_modes,
    get_active_fermionic_modes,
    get_fermionic_operator_types,
    predict_number_of_block_encoding_ancillae,
)


def _compute_n_elbows_and_anc_hw_per_incrementer(N, m):
    """
    |N⟩ -> |N + m⟩
    Returns:
        number of elbows and max clean ancillae to implement this operation
    """
    control = [cirq.LineQubit(0)]
    clean_ancillae = [cirq.LineQubit(i) for i in range(-1, -100, -1)]

    main_register = [cirq.LineQubit(i) for i in range(1, N + 1)]
    _, metrics = add_classical_value(
        main_register, m, clean_ancillae=clean_ancillae, ctrls=(control, [1])
    )
    return metrics.number_of_elbows, metrics.ancillae_highwater()


def count_metrics_analytic(operator, max_occupancy: int = 1):
    """
    For a given operator, with some max_occupancy for bosonic modes,
    returns analytic gate counts for the cost to implement the block encoding
    circuit, without compiling the cirq circuit.
    """

    if isinstance(operator, ParticleOperator):
        antifermions_present = operator.has_antifermions
        groups = operator.group()
    elif isinstance(operator, list):
        groups = operator
        operator = sum(operator, ParticleOperator())
        antifermions_present = operator.has_antifermions

    if antifermions_present:
        translated_groups = []
        max_fermionic_mode = operator.max_fermionic_mode
        for term in groups:
            translated_groups.append(
                translate_antifermions_to_fermions(term, max_fermionic_mode + 1)
            )
        groups = translated_groups

    L = len(groups)
    number_of_indexing_clean_ancillae = np.ceil(np.log2(L))

    metrics = CircuitMetrics()

    B = 0
    rescaling_factor = 0

    for term in groups:
        B = max(predict_number_of_block_encoding_ancillae(term), B)

        if len(term) == 1:
            active_fermionic_modes = get_active_fermionic_modes(term)
            active_bosonic_modes, exponents_list = get_bosonic_exponents(term)
            P = sum([exponents[0] + exponents[1] for exponents in exponents_list])

            if term.has_fermions and not term.has_bosons:  # bi^ bi, bi^ bi bj^ bj, ...
                rescaling_factor += 1 * np.abs(term.coeffs[0])
                number_of_elbows = len(active_fermionic_modes)
                clean_ancillae_usage = [
                    len(active_fermionic_modes) + number_of_indexing_clean_ancillae
                ]
                rotation_angles = []
            elif (
                term.has_bosons and not term.has_fermions
            ):  # ai^ ai, ai^ ai aj^ aj, ...
                rescaling_factor += np.abs(term.coeffs[0]) * max_occupancy ** (P / 2)
                number_of_elbows = int(np.ceil(np.log2(max_occupancy + 1))) * len(
                    (active_bosonic_modes)
                )
                clean_ancillae_usage = [
                    i + number_of_indexing_clean_ancillae
                    for i in range(1, int(np.ceil(np.log2(max_occupancy + 1))) + 1)
                ]
                rotation_angles = []
                for exponents in exponents_list:
                    angles = _get_bosonic_rotation_angles(
                        maximum_occupation_number=max_occupancy,
                        creation_exponent=exponents[0],
                        annihilation_exponent=exponents[1],
                    )
                    angles = np.concatenate(
                        [
                            angles,
                            np.zeros(
                                (1 << int(np.ceil(np.log2(len(angles))))) - len(angles)
                            ),
                        ]
                    )
                    processed_angles = list(_process_rotation_angles(angles))
                    rotation_angles += processed_angles + [
                        -sum(processed_angles) / 2,
                        sum(processed_angles) / 2,
                    ]

            else:  # bi^ bi ai^ ai
                rescaling_factor += np.abs(term.coeffs[0]) * max_occupancy ** (P / 2)
                clean_ancillae_usage = [
                    len(active_fermionic_modes) + number_of_indexing_clean_ancillae
                ] + [
                    i + number_of_indexing_clean_ancillae
                    for i in range(1, int(np.ceil(np.log2(max_occupancy + 1))) + 1)
                ]
                rotation_angles = []
                for exponents in exponents_list:
                    angles = _get_bosonic_rotation_angles(
                        maximum_occupation_number=max_occupancy,
                        creation_exponent=exponents[0],
                        annihilation_exponent=exponents[1],
                    )
                    angles = np.concatenate(
                        [
                            angles,
                            np.zeros(
                                (1 << int(np.ceil(np.log2(len(angles))))) - len(angles)
                            ),
                        ]
                    )
                    processed_angles = list(_process_rotation_angles(angles))
                    rotation_angles += processed_angles + [
                        -sum(processed_angles) / 2,
                        sum(processed_angles) / 2,
                    ]

                number_of_elbows = (len(active_fermionic_modes)) + int(
                    np.ceil(np.log2(max_occupancy + 1))
                ) * len(active_bosonic_modes)
        elif len(term) > 1:
            # term + h.c.
            term = term.to_list()[0]

            active_fermionic_modes = get_active_fermionic_modes(term)
            active_bosonic_modes = get_active_bosonic_modes(term)

            # Determine rotations
            _, exponents_list = get_bosonic_exponents(term, term.max_mode + 1)
            P = sum(
                [
                    exponents_list[i][0] + exponents_list[i][1]
                    for i in range(len(exponents_list))
                ]
            )
            RS_list = [
                exponents_list[i][0] - exponents_list[i][1]
                for i in range(len(exponents_list))
            ]

            rotation_angles = []
            for exponents in exponents_list:
                angles = _get_bosonic_rotation_angles(
                    maximum_occupation_number=max_occupancy,
                    creation_exponent=exponents[0],
                    annihilation_exponent=exponents[1],
                )
                angles = np.concatenate(
                    [
                        angles,
                        np.zeros(
                            (1 << int(np.ceil(np.log2(len(angles))))) - len(angles)
                        ),
                    ]
                )
                processed_angles = list(_process_rotation_angles(angles))
                rotation_angles += processed_angles + [
                    -sum(processed_angles) / 2,
                    sum(processed_angles) / 2,
                ]

            if term.has_fermions and not term.has_bosons:  # e.g. bi^ bj + bj^ bi
                rescaling_factor += np.abs(term.coeffs[0]) * 1
                clean_ancillae_usage = [
                    len(active_fermionic_modes) - 1 + number_of_indexing_clean_ancillae
                ]
                number_of_elbows = len(active_fermionic_modes) - 1

            elif term.has_bosons and not term.has_fermions:  # e.g. ai^ aj + aj^ ai
                rescaling_factor += np.abs(term.coeffs[0]) * (max_occupancy ** (P / 2))
                clean_ancillae_usage = [
                    i + number_of_indexing_clean_ancillae
                    for i in range(1, int(np.ceil(np.log2(max_occupancy + 1))) + 1 + 1)
                ]

                number_of_elbows = 1 + len(active_bosonic_modes) * int(
                    np.ceil(np.log2(max_occupancy + 1))
                )

                for RminusS in RS_list:  # +R - S
                    elbows, clean_anc = _compute_n_elbows_and_anc_hw_per_incrementer(
                        int(np.ceil(np.log2(max_occupancy + 1))), RminusS
                    )
                    number_of_elbows += elbows
                    clean_ancillae_usage.append(clean_anc)
                    elbows, clean_anc = _compute_n_elbows_and_anc_hw_per_incrementer(
                        int(np.ceil(np.log2(max_occupancy + 1))), -RminusS
                    )
                    number_of_elbows += elbows
                    clean_ancillae_usage.append(clean_anc)

            else:  # e.g. bi^ bj ak + h.c.
                rescaling_factor += np.abs(term.coeffs[0]) * max_occupancy ** (P / 2)
                clean_ancillae_usage = [
                    i + number_of_indexing_clean_ancillae
                    for i in range(1, int(np.ceil(np.log2(max_occupancy + 1))) + 1 + 1)
                ]
                number_of_elbows = (len(active_fermionic_modes)) + len(
                    active_bosonic_modes
                ) * int(np.ceil(np.log2(max_occupancy + 1)))
                for RminusS in RS_list:  # +R - S
                    elbows, clean_anc = _compute_n_elbows_and_anc_hw_per_incrementer(
                        int(np.ceil(np.log2(max_occupancy + 1))), RminusS
                    )
                    number_of_elbows += elbows
                    clean_ancillae_usage.append(clean_anc)
                    elbows, clean_anc = _compute_n_elbows_and_anc_hw_per_incrementer(
                        int(np.ceil(np.log2(max_occupancy + 1))), -RminusS
                    )
                    number_of_elbows += elbows
                    clean_ancillae_usage.append(clean_anc)

        metrics.number_of_elbows += number_of_elbows
        metrics.clean_ancillae_usage += clean_ancillae_usage
        metrics.rotation_angles += rotation_angles

    number_of_be_ancillae = np.ceil(np.log2(L)) + B
    metrics.number_of_elbows += L - 1  # number of left elbows from indexing

    return metrics, rescaling_factor, number_of_be_ancillae


from .interaction import _determine_block_encoding_function
from .index import index_over_terms
from .rescale import rescale_coefficients
from .system import System


def count_metrics_numeric(
    operator, max_bosonic_occupancy: int = 1, max_fermionic_mode=0
):
    groups = operator.group()
    assert len(groups) > 1

    if operator.has_antifermions:
        translated_groups = []
        for term in groups:
            translated_groups.append(
                translate_antifermions_to_fermions(term, max_fermionic_mode + 1)
            )
        groups = translated_groups
        operator = sum(groups, ParticleOperator())

    number_of_block_encoding_anillae = max(
        [predict_number_of_block_encoding_ancillae(group) for group in groups]
    )
    index_register = [
        cirq.LineQubit(-i - 2) for i in range(int(np.ceil(np.log2(len(groups)))))
    ]
    block_encoding_ancillae = [
        cirq.LineQubit(-10000 - i - len(index_register))
        for i in range(number_of_block_encoding_anillae)
    ]
    ctrls = ([cirq.LineQubit(0)], [1])
    clean_ancillae = [cirq.LineQubit(i + 10000) for i in range(10000)]
    number_of_fermionic_modes = 0
    number_of_bosonic_modes = 0
    if operator.max_fermionic_mode is not None:
        number_of_fermionic_modes = operator.max_fermionic_mode + 1
    if operator.max_bosonic_mode is not None:
        number_of_bosonic_modes = operator.max_bosonic_mode + 1
    system = System(
        max_bosonic_occupancy,
        10000,
        number_of_fermionic_modes=number_of_fermionic_modes,
        number_of_bosonic_modes=number_of_bosonic_modes,
    )

    block_encoding_functions = []
    rescaling_factors = []
    coefficients = []
    for term in groups:
        be_func, rescaling_factor = _determine_block_encoding_function(
            term, system, block_encoding_ancillae, clean_ancillae=clean_ancillae
        )
        block_encoding_functions.append(be_func)
        rescaling_factors.append(rescaling_factor)
        coefficients.append(np.abs(term.coeffs[0]))

    _, overall_rescaling_factor = rescale_coefficients(
        coefficients,
        rescaling_factors,
    )

    metrics = CircuitMetrics()

    _, _metrics = index_over_terms(
        index_register,
        block_encoding_functions,
        clean_ancillae=clean_ancillae,
        ctrls=ctrls,
    )

    metrics += _metrics

    L = len(groups)

    number_of_be_ancillae = np.ceil(np.log2(L)) + number_of_block_encoding_anillae

    return metrics, overall_rescaling_factor, number_of_be_ancillae


def _count_metrics_numeric_by_group(
    operator, max_bosonic_occupancy: int = 1, max_fermionic_mode=0
):
    if not operator.is_hermitian:
        operator = (operator + operator.dagger()).normal_order()
    if operator.has_antifermions:
        operator = translate_antifermions_to_fermions(operator, max_fermionic_mode + 1)

    number_of_block_encoding_anillae = predict_number_of_block_encoding_ancillae(
        operator
    )
    block_encoding_ancillae = [
        cirq.LineQubit(-10000 - i) for i in range(number_of_block_encoding_anillae)
    ]
    ctrls = ([cirq.LineQubit(0)], [1])
    clean_ancillae = [cirq.LineQubit(i + 10000) for i in range(10000)]
    number_of_fermionic_modes = 0
    number_of_bosonic_modes = 0
    if operator.max_fermionic_mode is not None:
        number_of_fermionic_modes = operator.max_fermionic_mode + 1
    if operator.max_bosonic_mode is not None:
        number_of_bosonic_modes = operator.max_bosonic_mode + 1
    system = System(
        max_bosonic_occupancy,
        100000,
        number_of_fermionic_modes=number_of_fermionic_modes,
        number_of_bosonic_modes=number_of_bosonic_modes,
    )

    be_func, rescaling_factor = _determine_block_encoding_function(
        operator,
        system,
        block_encoding_ancillae,
        clean_ancillae=clean_ancillae,
    )
    rescaling_factor *= np.abs(operator.coeffs[0])
    gates, metrics = be_func(ctrls=ctrls)

    return metrics, rescaling_factor, number_of_block_encoding_anillae, gates


def _count_metrics_analytic_fast(
    list_of_semi_groups, max_occupancy: int = 1, max_fermionic_mode: int = 0,
    in_parallel: bool = False
):

    translated_groups = []
    for term in list_of_semi_groups:
        translated_groups.append(
            translate_antifermions_to_fermions(term, max_fermionic_mode + 1)
        )
    groups = translated_groups

    L = len(groups)
    if in_parallel:
        L += 1
    number_of_indexing_clean_ancillae = np.ceil(np.log2(L))

    metrics = CircuitMetrics()

    B = 0
    rescaling_factor = 0

    for term in groups:
        if (
            term.normal_order().op_dict == {}
        ):  # Removes terms like bi^ bi^ which LOBE doesn't like
            continue

        if term.is_hermitian:
            B = max(predict_number_of_block_encoding_ancillae(term), B)
            active_fermionic_modes = get_active_fermionic_modes(term)
            active_bosonic_modes, exponents_list = get_bosonic_exponents(term)
            P = sum([exponents[0] + exponents[1] for exponents in exponents_list])

            if term.has_fermions and not term.has_bosons:  # bi^ bi, bi^ bi bj^ bj, ...
                rescaling_factor += 1 * np.abs(term.coeffs[0])
                number_of_elbows = len(active_fermionic_modes)
                clean_ancillae_usage = [
                    len(active_fermionic_modes) + number_of_indexing_clean_ancillae
                ]
                rotation_angles = []
            elif (
                term.has_bosons and not term.has_fermions
            ):  # ai^ ai, ai^ ai aj^ aj, ...
                rescaling_factor += np.abs(term.coeffs[0]) * max_occupancy ** (P / 2)
                number_of_elbows = int(np.ceil(np.log2(max_occupancy + 1))) * len(
                    (active_bosonic_modes)
                )
                clean_ancillae_usage = [
                    i + number_of_indexing_clean_ancillae
                    for i in range(1, int(np.ceil(np.log2(max_occupancy + 1))) + 1)
                ]
                rotation_angles = []
                for exponents in exponents_list:
                    angles = _get_bosonic_rotation_angles(
                        maximum_occupation_number=max_occupancy,
                        creation_exponent=exponents[0],
                        annihilation_exponent=exponents[1],
                    )
                    angles = np.concatenate(
                        [
                            angles,
                            np.zeros(
                                (1 << int(np.ceil(np.log2(len(angles))))) - len(angles)
                            ),
                        ]
                    )
                    processed_angles = list(_process_rotation_angles(angles))
                    rotation_angles += processed_angles + [
                        -sum(processed_angles) / 2,
                        sum(processed_angles) / 2,
                    ]

            else:  # bi^ bi ai^ ai
                rescaling_factor += np.abs(term.coeffs[0]) * max_occupancy ** (P / 2)
                clean_ancillae_usage = [
                    len(active_fermionic_modes) + number_of_indexing_clean_ancillae
                ] + [
                    i + number_of_indexing_clean_ancillae
                    for i in range(1, int(np.ceil(np.log2(max_occupancy + 1))) + 1)
                ]
                rotation_angles = []
                for exponents in exponents_list:
                    angles = _get_bosonic_rotation_angles(
                        maximum_occupation_number=max_occupancy,
                        creation_exponent=exponents[0],
                        annihilation_exponent=exponents[1],
                    )
                    angles = np.concatenate(
                        [
                            angles,
                            np.zeros(
                                (1 << int(np.ceil(np.log2(len(angles))))) - len(angles)
                            ),
                        ]
                    )
                    processed_angles = list(_process_rotation_angles(angles))
                    rotation_angles += processed_angles + [
                        -sum(processed_angles) / 2,
                        sum(processed_angles) / 2,
                    ]

                number_of_elbows = (len(active_fermionic_modes)) + int(
                    np.ceil(np.log2(max_occupancy + 1))
                ) * len(active_bosonic_modes)
        else:
            B = max(predict_number_of_block_encoding_ancillae(term + term.dagger()), B)
            # term + h.c.

            active_fermionic_modes, fermionic_operator_types = (
                get_fermionic_operator_types(term)
            )
            active_bosonic_modes = get_active_bosonic_modes(term)

            # Determine rotations
            _, exponents_list = get_bosonic_exponents(term, term.max_mode + 1)
            P = sum(
                [
                    exponents_list[i][0] + exponents_list[i][1]
                    for i in range(len(exponents_list))
                ]
            )
            RS_list = [
                exponents_list[i][0] - exponents_list[i][1]
                for i in range(len(exponents_list))
            ]

            rotation_angles = []
            for exponents in exponents_list:
                angles = _get_bosonic_rotation_angles(
                    maximum_occupation_number=max_occupancy,
                    creation_exponent=exponents[0],
                    annihilation_exponent=exponents[1],
                )
                angles = np.concatenate(
                    [
                        angles,
                        np.zeros(
                            (1 << int(np.ceil(np.log2(len(angles))))) - len(angles)
                        ),
                    ]
                )
                processed_angles = list(_process_rotation_angles(angles))
                rotation_angles += processed_angles + [
                    -sum(processed_angles) / 2,
                    sum(processed_angles) / 2,
                ]

            if term.has_fermions and not term.has_bosons:  # e.g. bi^ bj + bj^ bi
                rescaling_factor += np.abs(term.coeffs[0]) * 1
                clean_ancillae_usage = [
                    len(active_fermionic_modes) - 1 + number_of_indexing_clean_ancillae
                ]
                number_of_elbows = len(active_fermionic_modes) - 1

            elif term.has_bosons and not term.has_fermions:  # e.g. ai^ aj + aj^ ai
                rescaling_factor += np.abs(term.coeffs[0]) * (max_occupancy ** (P / 2))
                clean_ancillae_usage = [
                    i + number_of_indexing_clean_ancillae
                    for i in range(1, int(np.ceil(np.log2(max_occupancy + 1))) + 1 + 1)
                ]

                number_of_elbows = 1 + len(active_bosonic_modes) * int(
                    np.ceil(np.log2(max_occupancy + 1))
                )

                for RminusS in RS_list:  # +R - S
                    elbows, clean_anc = _compute_n_elbows_and_anc_hw_per_incrementer(
                        int(np.ceil(np.log2(max_occupancy + 1))), RminusS
                    )
                    number_of_elbows += elbows
                    clean_ancillae_usage.append(clean_anc)
                    elbows, clean_anc = _compute_n_elbows_and_anc_hw_per_incrementer(
                        int(np.ceil(np.log2(max_occupancy + 1))), -RminusS
                    )
                    number_of_elbows += elbows
                    clean_ancillae_usage.append(clean_anc)

            else:  # e.g. bi^ bj ak + h.c.
                rescaling_factor += np.abs(term.coeffs[0]) * max_occupancy ** (P / 2)
                clean_ancillae_usage = [
                    i + number_of_indexing_clean_ancillae
                    for i in range(1, int(np.ceil(np.log2(max_occupancy + 1))) + 1 + 1)
                ]
                number_of_elbows = (len(active_fermionic_modes)) + len(
                    active_bosonic_modes
                ) * int(np.ceil(np.log2(max_occupancy + 1)))
                for RminusS in RS_list:  # +R - S
                    elbows, clean_anc = _compute_n_elbows_and_anc_hw_per_incrementer(
                        int(np.ceil(np.log2(max_occupancy + 1))), RminusS
                    )
                    number_of_elbows += elbows
                    clean_ancillae_usage.append(clean_anc)
                    elbows, clean_anc = _compute_n_elbows_and_anc_hw_per_incrementer(
                        int(np.ceil(np.log2(max_occupancy + 1))), -RminusS
                    )
                    number_of_elbows += elbows
                    clean_ancillae_usage.append(clean_anc)

                if (np.all(np.array(fermionic_operator_types) == 2)) or (
                    np.all(np.array(fermionic_operator_types) == 3)
                ):
                    number_of_elbows += len(active_fermionic_modes)

        metrics.number_of_elbows += number_of_elbows
        metrics.clean_ancillae_usage += clean_ancillae_usage
        metrics.rotation_angles += rotation_angles

    number_of_be_ancillae = np.ceil(np.log2(L)) + B
    metrics.number_of_elbows += L - 1  # number of left elbows from indexing

    return (metrics, rescaling_factor, number_of_be_ancillae)
