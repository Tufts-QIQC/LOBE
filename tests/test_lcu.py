import pytest
import numpy as np
from src.lobe.lcu import LCU
from openparticle import ParticleOperator, generate_matrix
from src.lobe._utils import get_basis_of_full_system


@pytest.mark.parametrize("max_len_of_terms", [1, 2, 3])
@pytest.mark.parametrize("n_terms", [1, 2, 3])
@pytest.mark.parametrize("max_bosonic_occupancy", [1, 3])
def test_lcu_circuit_block_encodes_random_ParticleOperator(
    max_len_of_terms, n_terms, max_bosonic_occupancy
):
    op = ParticleOperator.random(
        n_terms=n_terms, max_len_of_terms=max_len_of_terms, max_mode=2
    )
    op.remove_identity()

    qubit_op = op.to_paulis(
        max_fermionic_mode=op.max_fermionic_mode,
        max_antifermionic_mode=op.max_antifermionic_mode,
        max_bosonic_mode=op.max_bosonic_mode,
        max_bosonic_occupancy=max_bosonic_occupancy,
    )

    lcu = LCU(op, max_bosonic_occupancy=max_bosonic_occupancy)

    if len(lcu.get_circuit().all_qubits()) >= 14:
        pytest.skip(
            f"too many qubits {len(lcu.get_circuit().all_qubits())} to explicitly validate"
        )
    assert np.allclose(lcu.unitary, qubit_op.to_sparse_matrix.toarray())


@pytest.mark.parametrize("max_bosonic_occupancy", [7, 15])
def test_lcu_circuit_block_encodes_bosonic_product(max_bosonic_occupancy):
    op = ParticleOperator("a0 a0")
    # op += ParticleOperator("a0^ a0^ a0^ a0^")

    full_fock_basis = get_basis_of_full_system(
        1,
        max_bosonic_occupancy,
        has_fermions=False,
        has_antifermions=False,
        has_bosons=True,
    )
    expected_unitary = generate_matrix(op, full_fock_basis)

    lcu = LCU(op, max_bosonic_occupancy=max_bosonic_occupancy)

    assert np.allclose(lcu.unitary, expected_unitary)
