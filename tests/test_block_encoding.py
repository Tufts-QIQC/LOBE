from openparticle import ParticleOperator
import numpy as np
import cirq
from src.lobe.system import System
from src.lobe.block_encoding import add_lobe_oracle
from src.lobe.usp import add_naive_usp
from src.lobe._utils import get_basis_of_full_system
import openparticle as op
import copy

# from cirq.contrib.svg import SVGCircuit


def test():
    maximum_occupation_number = 3
    max_number_of_bosonic_ops_in_term = 2
    terms = [
        # ParticleOperator("a0", 1),
        # ParticleOperator("a0^", 1),
        ParticleOperator("a0^ a1", 1),
        ParticleOperator("a1^ a0", 1),
    ]
    number_of_index_qubits = max(int(np.ceil(np.log2(len(terms)))), 1)
    max_coeff = max([term.coeff for term in terms])
    normalization_factor = (
        max_coeff
        * (1 << number_of_index_qubits)
        * ((maximum_occupation_number + 1) ** (max_number_of_bosonic_ops_in_term / 2))
    )
    normalized_terms = []
    for term in copy.deepcopy(terms):
        term.coeff /= 2
        normalized_terms.append(term)

    number_of_modes = max([mode for term in terms for mode in term.modes]) + 1
    number_of_ancillae = 6
    number_of_rotation_qubits = max_number_of_bosonic_ops_in_term + 1
    circuit = cirq.Circuit()
    validation = cirq.LineQubit(0)
    clean_ancillae = [cirq.LineQubit(i + 1) for i in range(number_of_ancillae)]
    rotation_qubits = [
        cirq.LineQubit(i + 1 + number_of_ancillae)
        for i in range(number_of_rotation_qubits)
    ]
    index_register = [
        cirq.LineQubit(i + 1 + number_of_ancillae + 3)
        for i in range(number_of_index_qubits)
    ]

    system = System(
        number_of_modes=number_of_modes,
        maximum_occupation_number=maximum_occupation_number,
        number_of_used_qubits=1 + number_of_ancillae + 3 + number_of_index_qubits,
        has_fermions=False,
        has_bosons=True,
    )
    circuit.append(
        cirq.I.on_each(validation, *clean_ancillae, *rotation_qubits, *index_register)
    )
    circuit += add_naive_usp(index_register)
    circuit.append(cirq.X.on(validation))
    circuit += add_lobe_oracle(
        terms,
        validation,
        index_register,
        system,
        rotation_qubits,
        clean_ancillae,
        perform_coefficient_oracle=True,
    )
    circuit += add_naive_usp(index_register)

    # with open("lobe.svg", "w") as f:
    #     f.write(SVGCircuit(circuit)._repr_svg_())

    upper_left_block = circuit.unitary(dtype=float)[
        : 1 << system.number_of_system_qubits, : 1 << system.number_of_system_qubits
    ]

    H = terms[0]
    if len(terms) > 0:
        for term in terms[1:]:
            H += term
    basis = get_basis_of_full_system(
        number_of_modes, maximum_occupation_number, has_bosons=True
    )
    hamiltonian_matrix = op.generate_matrix_from_basis(H, basis)

    assert np.allclose(hamiltonian_matrix, upper_left_block * normalization_factor)
