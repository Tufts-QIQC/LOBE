import pytest
import numpy as np
from src.lobe.lcu import LCU
from openparticle import ParticleOperator
from openparticle.qubit_mappings import op_qubit_map


@pytest.mark.parametrize("max_len_of_terms", [1, 2, 3])
@pytest.mark.parametrize("n_terms", [1, 2, 3])
@pytest.mark.parametrize("max_bose_occ", [1, 3])
def test_lcu_circuit_block_encodes_random_ParticleOperator(
    max_len_of_terms, n_terms, max_bose_occ
):
    op = ParticleOperator.random(
        n_terms=n_terms, max_len_of_terms=max_len_of_terms, max_mode=2
    )

    if () not in list(op.op_dict.keys()):
        qubit_op = op_qubit_map(op, max_bose_occ=max_bose_occ)

        lcu = LCU(op, max_bose_occ=max_bose_occ)

        if len(lcu.get_circuit().all_qubits()) >= 14:
            pytest.skip(
                f"too many qubits {len(lcu.all_qubits())} to explicitly validate"
            )
        assert np.allclose(lcu.unitary, qubit_op.to_sparse_matrix.toarray())
