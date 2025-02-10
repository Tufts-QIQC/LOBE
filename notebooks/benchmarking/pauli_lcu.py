from openparticle import ParticleOperator
import numpy as np
import cirq
import matplotlib.pyplot as plt

import sys, os

sys.path.append(os.path.join(os.path.dirname(os.path.realpath("__file__")), "../.."))
from src.lobe.system import System
from src.lobe.lcu import (
    pauli_lcu_block_encoding,
    estimate_pauli_lcu_rescaling_factor_and_number_of_be_ancillae,
    _select_paulis,
)
from src.lobe.lcu import LCU
from src.lobe._utils import _apply_negative_identity
from colors import *
from src.lobe.asp import get_target_state, add_prepare_circuit
from src.lobe.index import index_over_terms
from src.lobe.rescale import rescale_coefficients
from src.lobe.metrics import CircuitMetrics
from src.lobe.fermionic import fermionic_plus_hc_block_encoding
from tests._utils import _validate_block_encoding
from functools import partial


def lcuify(operator, max_bosonic_occupancy=1, zero_threshold=1e-6):
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

    rescaling_factor, number_of_block_encoding_ancillae = (
        estimate_pauli_lcu_rescaling_factor_and_number_of_be_ancillae(
            system, operator, zero_threshold=zero_threshold
        )
    )

    ctrls = ([cirq.LineQubit(0)], [1])
    index_register = [
        cirq.LineQubit(i + 1) for i in range(number_of_block_encoding_ancillae)
    ]
    clean_ancillae = [
        cirq.LineQubit(i + 100 + number_of_block_encoding_ancillae) for i in range(100)
    ]

    circuit = cirq.Circuit()
    circuit.append(cirq.X.on(ctrls[0][0]))

    system_register = system.fermionic_modes[::-1]
    for bosonic_reg in system.bosonic_modes[::-1]:
        system_register += bosonic_reg
    paulis = operator.to_paulis(
        max_fermionic_mode=operator.max_fermionic_mode,
        max_antifermionic_mode=operator.max_antifermionic_mode,
        max_bosonic_mode=operator.max_bosonic_mode,
        max_bosonic_occupancy=system.maximum_occupation_number,
        zero_threshold=zero_threshold,
    )
    gates, metrics = pauli_lcu_block_encoding(
        system,
        index_register,
        system_register,
        paulis,
        zero_threshold=zero_threshold,
        clean_ancillae=clean_ancillae,
        ctrls=ctrls,
    )
    circuit += gates
    circuit.append(cirq.X.on(ctrls[0][0]))

    _validate_block_encoding(
        circuit,
        system,
        rescaling_factor,
        operator,
        number_of_block_encoding_ancillae,
        max_bosonic_occupancy,
        max_qubits=16,
        using_pytest=False,
    )

    return (
        metrics,
        rescaling_factor,
        number_of_block_encoding_ancillae,
        system.number_of_system_qubits,
    )


from copy import deepcopy


def get_lcu_helper(
    operator,
    system,
    block_encoding_ancillae,
    clean_ancillae,
    zero_threshold=1e-6,
    max_fermionic_mode=None,
    max_antifermionic_mode=None,
    max_bosonic_mode=None,
):
    term = deepcopy(operator)
    term.op_dict[list(term.op_dict.keys())[0]] = 1
    term = ParticleOperator(term.op_dict)
    assert term.coeff == 1
    assert len(term.to_list()) == 1

    estimated_rescaling_factor = 1
    estimated_number_of_be_ancillae = 0
    for i, op in enumerate(term.split()[::-1]):
        _rescaling_factor, _number_used_be_ancillae = (
            estimate_pauli_lcu_rescaling_factor_and_number_of_be_ancillae(
                system, op, zero_threshold=zero_threshold
            )
        )
        estimated_rescaling_factor *= _rescaling_factor
        estimated_number_of_be_ancillae += _number_used_be_ancillae

    def _lcu_helper(ctrls=([], [])):
        _circuit = cirq.Circuit()
        _metrics = CircuitMetrics()
        overall_rescaling_factor = 1
        block_encoding_ancillae_counter = 0

        if np.isclose(np.sign(term.coeff), -1):
            _circuit += _apply_negative_identity(system.fermionic_modes[0], ctrls=ctrls)

        for i, op in enumerate(term.split()[::-1]):
            _rescaling_factor, number_of_block_encoding_ancillae = (
                estimate_pauli_lcu_rescaling_factor_and_number_of_be_ancillae(
                    system, op, zero_threshold=zero_threshold
                )
            )

            system_register = system.fermionic_modes[::-1]
            for bosonic_reg in system.bosonic_modes[::-1]:
                system_register += bosonic_reg
            paulis = op.to_paulis(
                max_fermionic_mode=max_fermionic_mode,
                max_antifermionic_mode=max_antifermionic_mode,
                max_bosonic_mode=max_bosonic_mode,
                max_bosonic_occupancy=system.maximum_occupation_number,
                zero_threshold=zero_threshold,
            )

            __gates, __metrics = pauli_lcu_block_encoding(
                system,
                block_encoding_ancillae[
                    block_encoding_ancillae_counter : block_encoding_ancillae_counter
                    + number_of_block_encoding_ancillae
                ],
                system_register,
                paulis,
                zero_threshold=zero_threshold,
                clean_ancillae=clean_ancillae,
                ctrls=ctrls,
            )
            block_encoding_ancillae_counter += number_of_block_encoding_ancillae

            _circuit += __gates
            _metrics += __metrics
            overall_rescaling_factor *= _rescaling_factor

        assert np.isclose(estimated_rescaling_factor, overall_rescaling_factor)
        return _circuit, _metrics

    return _lcu_helper, estimated_rescaling_factor, estimated_number_of_be_ancillae


def piecewise_lcu(operator, max_bosonic_occupancy=1, zero_threshold=1e-6):

    # max_number_of_be_ancillae = 0
    # for term in operator.to_list():
    #     _number_of_be_ancillae = 0
    #     for op in term.split():
    #         if op.particle_type == "fermionic":
    #             _number_of_be_ancillae += 1
    #         else:
    #             _number_of_be_ancillae += max_bosonic_occupancy
    #     max_number_of_be_ancillae = max(
    #         max_number_of_be_ancillae, _number_of_be_ancillae
    #     )

    ctrls = ([cirq.LineQubit(0)], [1])
    index_register = [
        cirq.LineQubit(i + 1)
        for i in range(max(int(np.ceil(np.log2(len(operator.to_list())))), 1))
    ]
    block_encoding_ancillae = [
        cirq.LineQubit(i + 1 + len(index_register)) for i in range(100)
    ]
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

    rescaling_factors, block_encoding_functions = [], []
    number_of_block_encoding_ancillae = 0
    for term in operator.to_list():
        be_function, rescaling_factor, used_be_ancillae = get_lcu_helper(
            term,
            system,
            block_encoding_ancillae,
            clean_ancillae=clean_ancillae[::-1],
            zero_threshold=zero_threshold,
            max_fermionic_mode=operator.max_fermionic_mode,
            max_antifermionic_mode=operator.max_antifermionic_mode,
            max_bosonic_mode=operator.max_bosonic_mode,
        )
        rescaling_factors.append(rescaling_factor)
        block_encoding_functions.append(be_function)
        number_of_block_encoding_ancillae = max(
            number_of_block_encoding_ancillae, used_be_ancillae
        )

    rescaled_coefficients, overall_rescaling_factor = rescale_coefficients(
        [term.coeffs[0] for term in operator.to_list()], rescaling_factors
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
        len(index_register) + number_of_block_encoding_ancillae,
        max_bosonic_occupancy,
        max_qubits=16,
        using_pytest=False,
    )

    return (
        metrics,
        overall_rescaling_factor,
        len(index_register) + number_of_block_encoding_ancillae,
        system.number_of_system_qubits,
    )
