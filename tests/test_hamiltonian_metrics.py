from src.lobe.hamiltonian_metrics import *
import numpy as np
import cirq
import pytest
from openparticle import ParticleOperator
from openparticle.hamiltonians.yukawa_hamiltonians import yukawa_hamiltonian
from src.lobe._utils import translate_antifermions_to_fermions

from src.lobe.interaction import _determine_block_encoding_function
from src.lobe.index import index_over_terms

def count_metrics_numeric(operator, max_bosonic_occupancy: int = 1):
    groups = operator.group()

    if operator.has_antifermions:
        translated_groups = []
        max_fermionic_mode = operator.max_fermionic_mode
        for term in groups:
            translated_groups.append(translate_antifermions_to_fermions(term, max_fermionic_mode))
        groups = translated_groups

    number_of_block_encoding_anillae = max(
        [predict_number_of_block_encoding_ancillae(group) for group in groups]
    )
    index_register = [
        cirq.LineQubit(-i - 2) for i in range(int(np.ceil(np.log2(len(groups)))))
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
    for term in groups:
        be_func, rescaling_factor = _determine_block_encoding_function(
            term, system, block_encoding_ancillae, clean_ancillae=clean_ancillae
        )
        block_encoding_functions.append(be_func)
        rescaling_factors.append(rescaling_factor)

    metrics = CircuitMetrics()


    _, _metrics = index_over_terms(
        index_register,
        block_encoding_functions,
        clean_ancillae=clean_ancillae,
        ctrls=ctrls,
    )
    metrics += _metrics
    
    metrics.rescaling_factor = sum(rescaling_factors)


    return metrics

@pytest.mark.parametrize('max_occupation', [1, 3])
def test_numeric_and_analytic_LOBE_counts(max_occupation):
    operator = yukawa_hamiltonian(2, 1, 1, 1)
    
    analytic_metrics = count_metrics(operator)
    numeric_metrics = count_metrics_numeric(operator)
    print("\n")
    print("---analytic---")
    analytic_metrics.display_metrics()
    print("\n")
    print("---numeric---")
    numeric_metrics.display_metrics()
    assert count_metrics(operator) == count_metrics_numeric(operator)

# def test_numeric_and_analytic_LOBE_counts_fermionic_only():
#     operator = ParticleOperator('b0^ b1^ b0 b1')
#     operator += operator.dagger()

#     analytic_metrics = count_metrics(operator)
#     numeric_metrics = count_metrics_numeric(operator)
#     print("\n")
#     print("---analytic---")
#     analytic_metrics.display_metrics()
#     print("\n")
#     print("---numeric---")
#     numeric_metrics.display_metrics()
#     assert count_metrics(operator) == count_metrics_numeric(operator)