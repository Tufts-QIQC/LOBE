from src.lobe.hamiltonian_metrics import (
    count_metrics_analytic,
)
import numpy as np
import cirq
import pytest
from openparticle import ParticleOperator
from openparticle.hamiltonians.yukawa_hamiltonians import yukawa_hamiltonian
from src.lobe._utils import (
    translate_antifermions_to_fermions,
    predict_number_of_block_encoding_ancillae,
)
from src.lobe.metrics import CircuitMetrics
from src.lobe.system import System
from src.lobe.interaction import _determine_block_encoding_function
from src.lobe.index import index_over_terms
from src.lobe.rescale import rescale_coefficients


def count_metrics_numeric(operator, max_bosonic_occupancy: int = 1):
    groups = operator.group()
    assert len(groups) > 1

    if operator.has_antifermions:
        translated_groups = []
        max_fermionic_mode = operator.max_fermionic_mode
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


def test_numeric_and_analytic_LOBE_counts_fermionic_product_of_number_ops():
    operator = ParticleOperator("b1^ b1 b2^ b2") + ParticleOperator("b0^ b0")

    analytic_metrics, analytic_rescaling_factor, analytic_n_be_anc = (
        count_metrics_analytic(operator)
    )

    numeric_metrics, numeric_rescaling_factor, numeric_n_be_anc = count_metrics_numeric(
        operator
    )

    assert analytic_metrics == numeric_metrics
    assert np.isclose(analytic_rescaling_factor, numeric_rescaling_factor)
    assert np.isclose(analytic_n_be_anc, numeric_n_be_anc)


@pytest.mark.parametrize("maximum_occupation", [1, 3, 7])
def test_numeric_and_analytic_LOBE_counts_bosonic_product_of_number_ops(
    maximum_occupation,
):
    operator = ParticleOperator(
        "a0^ a0 a1^ a1^ a1^ a1 a1 a1 a2^ a2 a3^ a3"
    ) + ParticleOperator("a0^ a0^ a0 a0")

    analytic_metrics, analytic_rescaling_factor, analytic_n_be_anc = (
        count_metrics_analytic(operator, maximum_occupation)
    )

    numeric_metrics, numeric_rescaling_factor, numeric_n_be_anc = count_metrics_numeric(
        operator, maximum_occupation
    )

    assert analytic_metrics == numeric_metrics
    assert np.isclose(analytic_rescaling_factor, numeric_rescaling_factor)
    assert np.isclose(analytic_n_be_anc, numeric_n_be_anc)


@pytest.mark.parametrize("maximum_occupation", [1, 3, 7])
def test_numeric_and_analytic_LOBE_counts_product_of_number_ops(maximum_occupation):
    operator = ParticleOperator("b0^ b0 a0^ a0") + ParticleOperator("a0^ a0 a1^ a1")

    analytic_metrics, analytic_rescaling_factor, analytic_n_be_anc = (
        count_metrics_analytic(operator, maximum_occupation)
    )

    numeric_metrics, numeric_rescaling_factor, numeric_n_be_anc = count_metrics_numeric(
        operator, maximum_occupation
    )

    assert analytic_metrics == numeric_metrics
    assert np.isclose(analytic_rescaling_factor, numeric_rescaling_factor)
    assert np.isclose(analytic_n_be_anc, numeric_n_be_anc)


def test_numeric_and_analytic_LOBE_counts_nondiagonal_fermion():
    operator = ParticleOperator("b0^ b1") + ParticleOperator("b0^ b1").dagger()
    operator += ParticleOperator("b2^ b3") + ParticleOperator("b2^ b3").dagger()
    operator += (
        ParticleOperator("b0^ b1 b0 b1^") + ParticleOperator("b0^ b1 b0 b1^").dagger()
    )

    analytic_metrics, analytic_rescaling_factor, analytic_n_be_anc = (
        count_metrics_analytic(operator)
    )

    numeric_metrics, numeric_rescaling_factor, numeric_n_be_anc = count_metrics_numeric(
        operator
    )

    assert analytic_metrics == numeric_metrics
    assert np.isclose(analytic_rescaling_factor, numeric_rescaling_factor)
    assert np.isclose(analytic_n_be_anc, numeric_n_be_anc)


@pytest.mark.parametrize("maximum_occupation", [1, 3, 7])
def test_numeric_and_analytic_LOBE_counts_nondiagonal_boson(maximum_occupation):
    operator = 2 * ParticleOperator("a0^ a1") + 2 * ParticleOperator("a0^ a1").dagger()
    operator += ParticleOperator("a2^ a3") + ParticleOperator("a2^ a3").dagger()
    operator += (
        ParticleOperator("a0^ a0^ a1 a1") + ParticleOperator("a0^ a0^ a1 a1").dagger()
    )

    analytic_metrics, analytic_rescaling_factor, analytic_n_be_anc = (
        count_metrics_analytic(operator, maximum_occupation)
    )

    numeric_metrics, numeric_rescaling_factor, numeric_n_be_anc = count_metrics_numeric(
        operator, maximum_occupation
    )

    assert analytic_metrics == numeric_metrics
    assert np.isclose(analytic_rescaling_factor, numeric_rescaling_factor)
    assert np.isclose(analytic_n_be_anc, numeric_n_be_anc)


@pytest.mark.parametrize("maximum_occupation", [1, 3, 7])
def test_numeric_and_analytic_LOBE_counts_interaction(maximum_occupation):
    operator = (
        0.15 * ParticleOperator("b1^ b0 a1")
        + 0.15 * ParticleOperator("b1^ b0 a1").dagger()
    )
    operator += ParticleOperator("b0^ b1 a0") + ParticleOperator("b0^ b1 a0").dagger()

    analytic_metrics, analytic_rescaling_factor, analytic_n_be_anc = (
        count_metrics_analytic(operator, maximum_occupation)
    )

    numeric_metrics, numeric_rescaling_factor, numeric_n_be_anc = count_metrics_numeric(
        operator, maximum_occupation
    )

    assert analytic_metrics == numeric_metrics
    assert np.isclose(analytic_rescaling_factor, numeric_rescaling_factor)
    assert np.isclose(analytic_n_be_anc, numeric_n_be_anc)


@pytest.mark.parametrize("maximum_occupation", [1, 3, 7])
def test_numeric_and_analytic_LOBE_counts_arbitrary_operator(maximum_occupation):
    operator = yukawa_hamiltonian(2, 1, 1, 1)
    operator += (
        ParticleOperator("b0^ b1^ b0 b1") + ParticleOperator("b0^ b1^ b0 b1").dagger()
    )
    operator += (
        ParticleOperator("a0^ a1^ a0 a1") + ParticleOperator("a0^ a1^ a0 a1").dagger()
    )

    analytic_metrics, analytic_rescaling_factor, analytic_n_be_anc = (
        count_metrics_analytic(operator, maximum_occupation)
    )

    numeric_metrics, numeric_rescaling_factor, numeric_n_be_anc = count_metrics_numeric(
        operator, maximum_occupation
    )

    assert analytic_metrics == numeric_metrics
    assert np.isclose(analytic_rescaling_factor, numeric_rescaling_factor)
    assert np.isclose(analytic_n_be_anc, numeric_n_be_anc)


def test_pregrouped_operator():
    maximum_occupation = 3
    operator = yukawa_hamiltonian(2, 1, 1, 1)
    operator += (
        ParticleOperator("b0^ b1^ b0 b1") + ParticleOperator("b0^ b1^ b0 b1").dagger()
    )
    operator += (
        ParticleOperator("a0^ a1^ a0 a1") + ParticleOperator("a0^ a1^ a0 a1").dagger()
    )
    numeric_metrics, numeric_rescaling_factor, numeric_n_be_anc = count_metrics_numeric(
        operator, maximum_occupation
    )
    operator = operator.group()
    analytic_metrics, analytic_rescaling_factor, analytic_n_be_anc = (
        count_metrics_analytic(operator, maximum_occupation)
    )

    assert analytic_metrics == numeric_metrics
    assert np.isclose(analytic_rescaling_factor, numeric_rescaling_factor)
    assert np.isclose(analytic_n_be_anc, numeric_n_be_anc)
