import pytest
from src.lobe.lcu import *


@pytest.mark.parametrize("n_terms", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("n_qubits", [1, 2, 3, 4, 5])
def test_lcu_circuit_block_encodes_random_PauliwordOp(n_terms, n_qubits):
    op = PauliwordOp.random(n_terms=n_terms, n_qubits=n_qubits)
    lcu = LCU(op)
    assert np.allclose(lcu.unitary, op.to_sparse_matrix.toarray())


@pytest.mark.parametrize("max_len_of_terms", [1, 2, 3])
@pytest.mark.parametrize("n_terms", [1, 2, 3])
@pytest.mark.parametrize("max_bose_occ", [1, 3])
def test_lcu_circuit_block_encodes_random_ParticleOperator(
    max_len_of_terms, n_terms, max_bose_occ
):
    op = ParticleOperator.random(
        n_terms=n_terms, max_len_of_terms=max_len_of_terms, max_mode=2
    )
    qubit_op = op_qubit_map(op, max_bose_occ=max_bose_occ)

    lcu = LCU(qubit_op)
    assert np.allclose(lcu.unitary, qubit_op.to_sparse_matrix.toarray())
