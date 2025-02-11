from openparticle.hamiltonians.yukawa_hamiltonians import yukawa_hamiltonian
from openparticle import ParticleOperator
import numpy as np
import cirq
import matplotlib.pyplot as plt

import sys, os

sys.path.append(os.path.join(os.path.dirname(os.path.realpath("__file__")), "../../.."))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath("__file__")), "../.."))
from src.lobe.asp import get_target_state, add_prepare_circuit
from src.lobe.rescale import get_number_of_active_bosonic_modes, rescale_coefficients
from src.lobe.system import System
from colors import *
from src.lobe.lcu import LCU
from lobe.yukawa import _determine_block_encoding_function
from src.lobe._utils import translate_antifermions_to_fermions
from src.lobe.index import index_over_terms
from src.lobe.metrics import CircuitMetrics
from tests._utils import _validate_block_encoding
from functools import partial
from openparticle import generate_matrix
from src.lobe._utils import get_basis_of_full_system


def lobotomize(operator, max_bosonic_occupancy):
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
    system = System(
        operator.max_mode + 1, max_bosonic_occupancy, 1000, True, False, True
    )

    block_encoding_functions = []
    rescaling_factors = []
    for term in terms:
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
        index_register, block_encoding_functions, clean_ancillae, ctrls=ctrls
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
        max_qubits=16,
        using_pytest=False,
    )

    return (
        metrics,
        overall_rescaling_factor,
        len(index_register) + number_of_block_encoding_anillae,
        system.number_of_system_qubits,
    )


def lcu_ify(operator, max_bosonic_occupancy):
    lcu = LCU(
        operator, max_bosonic_occupancy=max_bosonic_occupancy, zero_threshold=1e-6
    )
    ctrls = ([cirq.LineQubit(-1000000)], [1])
    circuit = cirq.Circuit()
    circuit += cirq.X.on(ctrls[0][0])
    circuit += lcu.get_circuit(ctrls=ctrls)
    circuit += cirq.X.on(ctrls[0][0])
    fake_sys = System(
        operator.max_mode + 1,
        max_bosonic_occupancy,
        1000 + lcu.number_of_index_qubits,
        operator.has_fermions,
        operator.has_antifermions,
        operator.has_bosons,
    )
    _validate_block_encoding(
        circuit,
        fake_sys,
        lcu.one_norm,
        operator,
        len(lcu.index_register),
        max_bosonic_occupancy,
        max_qubits=16,
        using_pytest=False,
    )

    return (
        lcu.circuit_metrics,
        lcu.one_norm,
        len(lcu.index_register),
        fake_sys.number_of_system_qubits,
    )
