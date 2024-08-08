from openparticle import ParticleOperator, FermionOperator, generate_matrix
import numpy as np

import cirq
from src.lobe.system import System
from src.lobe.block_encoding import add_lobe_oracle
from src.lobe.usp import add_naive_usp
from src.lobe._utils import get_basis_of_full_system
from src.lobe.lobe_circuit import lobe_circuit
from src.lobe.rescale import rescale_terms, get_numbers_of_bosonic_operators_in_terms
import pytest


def _test_helper(terms, maximum_occupation_number, decompose):
    hamiltonian = terms[0]
    for term in terms[1:]:
        hamiltonian += term

    number_of_modes = max([term.max_mode() for term in terms]) + 1

    full_fock_basis = get_basis_of_full_system(
        number_of_modes,
        maximum_occupation_number,
        has_fermions=hamiltonian.has_fermions,
        has_antifermions=hamiltonian.has_antifermions,
        has_bosons=hamiltonian.has_bosons,
    )
    matrix = generate_matrix(hamiltonian, full_fock_basis)

    rescaled_terms, scaling_factor = rescale_terms(terms, maximum_occupation_number)

    max_number_of_bosonic_ops_in_term = max(
        get_numbers_of_bosonic_operators_in_terms(terms)
    )

    number_of_ancillae = 5
    number_of_index_qubits = max(int(np.ceil(np.log2(len(terms)))), 1)
    number_of_rotation_qubits = max_number_of_bosonic_ops_in_term + 1

    block_encoding_scaling_factor = (1 << number_of_index_qubits) * scaling_factor

    # Declare Qubits
    circuit = cirq.Circuit()
    validation = cirq.LineQubit(0)
    clean_ancillae = [cirq.LineQubit(i + 1) for i in range(number_of_ancillae)]
    rotation_qubits = [
        cirq.LineQubit(i + 1 + number_of_ancillae)
        for i in range(number_of_rotation_qubits)
    ]
    index_register = [
        cirq.LineQubit(i + 1 + number_of_ancillae + number_of_rotation_qubits)
        for i in range(number_of_index_qubits)
    ]
    system = System(
        number_of_modes=number_of_modes,
        maximum_occupation_number=maximum_occupation_number,
        number_of_used_qubits=1
        + number_of_ancillae
        + number_of_rotation_qubits
        + number_of_index_qubits,
        has_fermions=hamiltonian.has_fermions,
        has_antifermions=hamiltonian.has_antifermions,
        has_bosons=hamiltonian.has_bosons,
    )
    circuit.append(cirq.I.on_each(*system.fermionic_register))
    circuit.append(cirq.I.on_each(*system.antifermionic_register))
    for bosonic_reg in system.bosonic_system:
        circuit.append(cirq.I.on_each(*bosonic_reg))

    # Generate full Block-Encoding circuit
    circuit.append(cirq.X.on(validation))
    circuit += add_naive_usp(index_register)
    circuit += add_lobe_oracle(
        rescaled_terms,
        validation,
        index_register,
        system,
        rotation_qubits,
        clean_ancillae,
        perform_coefficient_oracle=True,
        decompose=decompose,
    )
    circuit += add_naive_usp(index_register)

    upper_left_block = circuit.unitary(dtype=complex)[
        : 1 << system.number_of_system_qubits, : 1 << system.number_of_system_qubits
    ]

    assert np.allclose(upper_left_block * block_encoding_scaling_factor, matrix)


@pytest.mark.parametrize(
    "terms",
    [
        [
            ParticleOperator("a0^ a0"),
        ],
        [
            ParticleOperator("b0^ b0"),
        ],
        [
            ParticleOperator("b1^ b1"),
            ParticleOperator("b0^ b0"),
        ],
        [
            ParticleOperator("b0^ b0"),
            ParticleOperator("b0^ b1"),
            ParticleOperator("b1^ b0"),
            ParticleOperator("b1^ b1"),
        ],
        [
            ParticleOperator("d0^ d0"),
            ParticleOperator("d0^ d1"),
            ParticleOperator("d1^ d0"),
            ParticleOperator("d1^ d1"),
        ],
        [
            ParticleOperator("d0^ d0"),
        ],
        [
            ParticleOperator("b1^ b1"),
            ParticleOperator("b0"),
        ],
        [
            ParticleOperator("d1^ d1"),
            ParticleOperator("d0"),
        ],
        [
            ParticleOperator("a0"),
            ParticleOperator("a1"),
            ParticleOperator("a0^ a1"),
            ParticleOperator("a1^ a0"),
        ],
        [
            ParticleOperator("a0"),
            ParticleOperator("a1^ a0"),
        ],
        [
            ParticleOperator("d0^ a0^ b0"),
            -0.5 * ParticleOperator("b0^ d0"),
            0.25 * ParticleOperator("b0^ d0^ a0"),
            1 / 3 * ParticleOperator("a0^ d0"),
        ],
    ],
)
@pytest.mark.parametrize("maximum_occupation_number", [1, 3])
def test_lobe_block_encoding_undecomposed(
    terms,
    maximum_occupation_number,
):
    _test_helper(terms, maximum_occupation_number, False)


@pytest.mark.parametrize(
    "terms",
    [
        [
            ParticleOperator("a0"),
        ],
        [
            ParticleOperator("a0^"),
        ],
        [
            ParticleOperator("a0^ b0"),
            0.25 * ParticleOperator("b0^ a0"),
        ],
    ],
)
@pytest.mark.parametrize("maximum_occupation_number", [1, 3])
def test_lobe_block_encoding_decomposed(
    terms,
    maximum_occupation_number,
):
    _test_helper(terms, maximum_occupation_number, True)


@pytest.mark.parametrize(
    "terms, maximum_occupation_number",
    [
        (
            [
                ParticleOperator("a0^ a0^ a0^ a0^ a0^ a0^ a0 a0 a0 a0 a0 a0"),
            ],
            15,
        ),
        (
            [
                ParticleOperator("a0^ a0^ a0^ a0^ a0^ a0^ a0 a0 a0 a0 a0 a0"),
                ParticleOperator("a0^ a0^ a0^ a0 a0 a0 a0 a0"),
            ],
            15,
        ),
    ],
)
def test_lobe_block_encoding_large_occupancy(
    terms,
    maximum_occupation_number,
):
    _test_helper(terms, maximum_occupation_number, False)
