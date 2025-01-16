import pytest
import numpy as np
from src.lobe.lcu import LCU
from openparticle import ParticleOperator, generate_matrix, get_fock_basis
from src.lobe._utils import get_basis_of_full_system


@pytest.mark.parametrize(
    "op",
    [
        ParticleOperator("b0"),
        ParticleOperator("a0 a0"),
        ParticleOperator("a0 a1"),
        ParticleOperator("a0") + ParticleOperator("a2"),
        ParticleOperator("a0 a0") + ParticleOperator("a0^ a0^"),
        ParticleOperator("b0 b1"),
        ParticleOperator("b0 d0"),
        ParticleOperator("b1 d1"),
        ParticleOperator("b0 b1 d0"),
        # ParticleOperator("b0 b2 b3 d2"),
    ],
)
def test_lcu_circuit_block_encodes_operator(op):
    max_bosonic_occupancy = 3

    full_fock_basis = get_fock_basis(op, max_bosonic_occupancy)
    expected_unitary = generate_matrix(op, full_fock_basis)

    lcu = LCU(op, max_bosonic_occupancy=max_bosonic_occupancy)

    assert np.allclose(lcu.unitary, expected_unitary)
