from openparticle.utils import get_fock_basis, generate_matrix
from openparticle.hamiltonians.phi4_hamiltonian import phi4_Hamiltonian
import cirq
import matplotlib.pyplot as plt
import numpy as np
from functools import partial
import sys, os

sys.path.append(
    os.path.join(os.path.dirname(os.path.realpath("__file__")), "../../../")
)
sys.path.append(os.path.join(os.path.dirname(os.path.realpath("__file__")), "../../"))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath("__file__")), "../"))

from src.lobe.system import System
from pauli_lcu import lcuify, piecewise_lcu
from src.lobe.asp import get_target_state, add_prepare_circuit
from src.lobe.rescale import (
    get_number_of_active_bosonic_modes,
)
from src.lobe._utils import get_bosonic_exponents
from src.lobe.index import index_over_terms
from src.lobe.metrics import CircuitMetrics
from src.lobe.bosonic import (
    bosonic_product_plus_hc_block_encoding,
    bosonic_product_block_encoding,
)
from tests._utils import _validate_block_encoding

from colors import *


def _blank_be_func(ctrls=([], [])):
    return [], CircuitMetrics()


def lobeify(operator, maximum_occupation_number):

    grouped_terms = operator.group()

    number_of_block_encoding_ancillae = max(
        get_number_of_active_bosonic_modes(grouped_terms)
    )
    if len(grouped_terms) != len(operator.to_list()):
        number_of_block_encoding_ancillae += 1  # For grouped terms, need an additional BE ancilla to index between the group

    index_register = [
        cirq.LineQubit(-i - 2) for i in range(int(np.ceil(np.log2(len(grouped_terms)))))
    ]
    block_encoding_ancillae = [
        cirq.LineQubit(-100 - i - len(index_register))
        for i in range(number_of_block_encoding_ancillae)
    ]

    ctrls = ([cirq.LineQubit(0)], [1])
    clean_ancillae = [cirq.LineQubit(i + 100) for i in range(100)]
    number_of_fermionic_modes = 0
    number_of_bosonic_modes = 0
    if operator.max_fermionic_mode is not None:
        number_of_fermionic_modes = operator.max_fermionic_mode + 1
    if operator.max_bosonic_mode is not None:
        number_of_bosonic_modes = operator.max_bosonic_mode + 1
    system = System(
        maximum_occupation_number,
        1000,
        number_of_fermionic_modes=number_of_fermionic_modes,
        number_of_bosonic_modes=number_of_bosonic_modes,
    )

    block_encoding_functions = []
    rescaling_factors = []
    for term in grouped_terms:

        if term.has_identity():
            block_encoding_functions.append(_blank_be_func)
            rescaling_factors.append(1)
            continue

        plus_hc = False
        if len(term) == 2:
            plus_hc = True
            term = term.to_list()[0]
        active_modes, exponents = get_bosonic_exponents(
            term, operator.max_bosonic_mode + 1
        )

        if term.has_identity():
            pass
        elif not plus_hc:
            block_encoding_functions.append(
                partial(
                    bosonic_product_block_encoding,
                    system=system,
                    block_encoding_ancillae=block_encoding_ancillae,
                    active_indices=active_modes,
                    exponents_list=exponents,
                    clean_ancillae=clean_ancillae[1:],
                )
            )
            rescaling_factors.append(
                np.sqrt(maximum_occupation_number) ** (sum(sum(np.asarray(exponents))))
            )
        else:
            block_encoding_functions.append(
                partial(
                    bosonic_product_plus_hc_block_encoding,
                    system=system,
                    block_encoding_ancillae=block_encoding_ancillae,
                    active_indices=active_modes,
                    exponents_list=exponents,
                    clean_ancillae=clean_ancillae[1:],
                )
            )
            rescaling_factors.append(
                2
                * np.sqrt(maximum_occupation_number)
                ** (sum(sum(np.asarray(exponents))))
            )

    rescaled_coefficients = []
    for term, rescaling_factor in zip(grouped_terms, rescaling_factors):
        rescaled_coefficients.append(
            term.coeffs[0] * rescaling_factor / max(rescaling_factors)
        )

    target_state = get_target_state(rescaled_coefficients)
    gates = []
    metrics = CircuitMetrics()
    for mode in system.bosonic_modes:
        for qubit in mode:
            gates.append(cirq.I.on(qubit))

    gates.append(cirq.X.on(ctrls[0][0]))

    _gates, _metrics = add_prepare_circuit(
        index_register, target_state, clean_ancillae=clean_ancillae
    )
    print("Metrics from add_prepare_circuit: \n")
    _metrics.display_metrics()
    gates += _gates
    metrics += _metrics

    _gates, _metrics = index_over_terms(
        index_register, block_encoding_functions, clean_ancillae, ctrls=ctrls
    )
    print("Metrics from index_over_terms: \n")
    _metrics.display_metrics()  # Print metrics
    gates += _gates
    metrics += _metrics

    _gates, _metrics = add_prepare_circuit(
        index_register, target_state, dagger=True, clean_ancillae=clean_ancillae
    )
    print("Metrics from add_prepare_circuit: \n")
    _metrics.display_metrics()
    gates += _gates
    metrics += _metrics

    gates.append(cirq.X.on(ctrls[0][0]))

    overall_rescaling_factor = sum(
        [
            term.coeffs[0] * rescaling_factor
            for term, rescaling_factor in zip(grouped_terms, rescaling_factors)
        ]
    )
    print("Total metrics of the block encoding: \n")
    metrics.display_metrics()
    _validate_block_encoding(
        cirq.Circuit(gates),
        system,
        overall_rescaling_factor,
        operator,
        len(index_register) + number_of_block_encoding_ancillae,
        maximum_occupation_number,
        using_pytest=False,
    )

    return (
        metrics,
        overall_rescaling_factor,
        len(index_register) + number_of_block_encoding_ancillae,
        system.number_of_system_qubits,
    )


def _get_phi4_hamiltonian_norm(res, maximum_occupation_number, g=1):
    ham = phi4_Hamiltonian(res, g=g, mb=1)
    basis = get_fock_basis(ham, maximum_occupation_number)
    matrix = generate_matrix(ham, basis)

    return np.linalg.norm(matrix, ord=2)


def get_data(omega, resolutions):
    LCU_DATA = []
    LCU_PIECEWISE_DATA = []
    LOBE_DATA = []
    operator_norms = []
    for resolution in resolutions:
        print("---", resolution, "---", omega, "---")
        operator = phi4_Hamiltonian(resolution, 1, 1).normal_order()

        LCU_DATA.append(lcuify(operator, omega))
        LCU_PIECEWISE_DATA.append(piecewise_lcu(operator, omega))
        LOBE_DATA.append(lobeify(operator, omega))
        operator_norms.append(_get_phi4_hamiltonian_norm(resolution, omega))

    return LCU_DATA, LCU_PIECEWISE_DATA, LOBE_DATA, operator_norms


resolution_range = np.arange(2, 8, 1)
omega = 3
LCU_DATA, LCU_PIECEWISE_DATA, LOBE_DATA, operator_norms = get_data(
    omega, resolution_range
)

import pickle

with open(
    f"phi4_data/phi4_{omega}_{resolution_range[0]}_{resolution_range[-1]}.pickle", "wb"
) as handle:
    pickle.dump(
        (
            LCU_DATA,
            LCU_PIECEWISE_DATA,
            LOBE_DATA,
            operator_norms,
            omega,
            resolution_range,
        ),
        handle,
        protocol=pickle.HIGHEST_PROTOCOL,
    )
