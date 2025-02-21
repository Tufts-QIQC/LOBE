from openparticle.hamiltonians.yukawa_hamiltonians import yukawa_hamiltonian
import numpy as np
import cirq
import sys, os

sys.path.append(os.path.join(os.path.dirname(os.path.realpath("__file__")), "../../.."))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath("__file__")), "../.."))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath("__file__")), "../"))
from pauli_lcu import lcuify, piecewise_lcu
from src.lobe.asp import get_target_state, add_prepare_circuit
from src.lobe.rescale import rescale_coefficients
from src.lobe.system import System
from colors import *
from src.lobe.yukawa import _determine_block_encoding_function
from src.lobe._utils import translate_antifermions_to_fermions
from src.lobe.index import index_over_terms
from src.lobe.metrics import CircuitMetrics
from tests._utils import _validate_block_encoding
from openparticle import generate_matrix
from src.lobe._utils import get_basis_of_full_system


def _blank_be_func(ctrls=([], [])):
    return [], CircuitMetrics()


def lobeify(operator, max_bosonic_occupancy):
    terms = operator.group()

    number_of_block_encoding_anillae = 3
    index_register = [
        cirq.LineQubit(-i - 2) for i in range(int(np.ceil(np.log2(len(terms)))))
    ]
    block_encoding_ancillae = [
        cirq.LineQubit(-100 - i - len(index_register))
        for i in range(number_of_block_encoding_anillae)
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
        max_bosonic_occupancy,
        1000,
        number_of_fermionic_modes=number_of_fermionic_modes,
        number_of_bosonic_modes=number_of_bosonic_modes,
    )

    block_encoding_functions = []
    rescaling_factors = []
    for term in terms:

        if term.has_identity():
            block_encoding_functions.append(_blank_be_func)
            rescaling_factors.append(1)
            continue

        be_func, rescaling_factor = _determine_block_encoding_function(
            term, system, block_encoding_ancillae, clean_ancillae=clean_ancillae
        )
        block_encoding_functions.append(be_func)
        rescaling_factors.append(rescaling_factor)

    rescaled_coefficients, overall_rescaling_factor = rescale_coefficients(
        [term.coeffs[0] for term in terms], rescaling_factors
    )
    target_state = get_target_state(rescaled_coefficients)

    # Generate Circuit
    gates = []
    metrics = CircuitMetrics()

    gates.append(cirq.X.on(ctrls[0][0]))
    _gates, _metrics = add_prepare_circuit(
        index_register, target_state, clean_ancillae=clean_ancillae
    )
    gates += _gates
    metrics += _metrics

    _gates, _metrics = index_over_terms(
        index_register,
        block_encoding_functions,
        clean_ancillae=clean_ancillae,
        ctrls=ctrls,
    )
    gates += _gates
    metrics += _metrics

    _gates, _metrics = add_prepare_circuit(
        index_register, target_state, dagger=True, clean_ancillae=clean_ancillae
    )
    gates += _gates
    metrics += _metrics
    gates.append(cirq.X.on(ctrls[0][0]))

    circuit = cirq.Circuit(gates)
    _validate_block_encoding(
        circuit,
        system,
        overall_rescaling_factor,
        operator,
        len(index_register) + number_of_block_encoding_anillae,
        max_bosonic_occupancy,
        max_qubits=22,
        using_pytest=False,
    )

    return (
        metrics,
        overall_rescaling_factor,
        len(index_register) + number_of_block_encoding_anillae,
        system.number_of_system_qubits,
    )


def _get_hamiltonian_norm(operator, maximum_occupation_number):
    basis = get_basis_of_full_system(
        maximum_occupation_number,
        operator.max_fermionic_mode + 1,
        operator.max_bosonic_mode + 1,
    )
    matrix = generate_matrix(operator, basis)

    return np.linalg.norm(matrix, ord=2)


def get_phi4_data_changing_resolution(omega, resolutions):
    LCU_DATA = []
    LCU_PIECEWISE_DATA = []
    LOBE_DATA = []
    operator_norms = []
    for resolution in resolutions:
        print("---", resolution, "---", omega, "---")
        operator = yukawa_hamiltonian(res=resolution, g=1, mf=1, mb=1)
        operator = translate_antifermions_to_fermions(operator).normal_order()

        LCU_DATA.append(lcuify(operator, omega))
        LCU_PIECEWISE_DATA.append(piecewise_lcu(operator, omega))
        LOBE_DATA.append(lobeify(operator, omega))
        operator_norms.append(_get_hamiltonian_norm(operator, omega))

    return LCU_DATA, LCU_PIECEWISE_DATA, LOBE_DATA, operator_norms


resolution_range = np.arange(2, 7, 1)
omega = 3
LCU_DATA, LCU_PIECEWISE_DATA, LOBE_DATA, operator_norms = (
    get_phi4_data_changing_resolution(omega, resolution_range)
)

import pickle

with open(
    f"full_yukawa_data/full_yukawa_{omega}_{resolution_range[0]}_{resolution_range[-1]}.pickle",
    "wb",
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
