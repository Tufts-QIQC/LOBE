from openparticle import *
import numpy as np
import cirq
from .system import System
from .block_encoding import add_lobe_oracle
from .usp import add_naive_usp
from .rescale import rescale_terms, get_number_of_active_bosonic_modes
from typing import Union, List


def lobe_circuit(
    operator: Union[ParticleOperator, List],
    max_bose_occ: int = 1,
    decompose: bool = False,
    return_unitary: bool = True,
    return_matrix: bool = False,
    return_numerics: bool = False,
):
    NUMERICS = {
        "left_elbows": 0,
        "right_elbows": 0,
        "rotations": 0,
        "ancillae_tracker": [0],
        "angles": [],
    }

    if isinstance(operator, List):
        terms = operator
        op_as_po = ParticleOperator({})
        for op in operator:
            op_as_po += op
        operator = op_as_po
    elif isinstance(operator, ParticleOperator):
        terms = operator.to_list()

    number_of_modes = max([term.max_mode() for term in terms]) + 1

    rescaled_terms, scaling_factor = rescale_terms(terms, max_bose_occ)

    number_of_ancillae = (
        1000  # Some arbitrary large number with most ancilla disregarded
    )
    number_of_index_qubits = max(int(np.ceil(np.log2(len(terms)))), 1)
    number_of_rotation_qubits = max(get_number_of_active_bosonic_modes(terms)) + 1

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
        maximum_occupation_number=max_bose_occ,
        number_of_used_qubits=1
        + number_of_ancillae
        + number_of_rotation_qubits
        + number_of_index_qubits,
        has_fermions=operator.has_fermions,
        has_antifermions=operator.has_antifermions,
        has_bosons=operator.has_bosons,
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
        numerics=NUMERICS,
    )
    circuit += add_naive_usp(index_register)

    unitary = None
    matrix = None

    if return_unitary:
        if return_matrix:
            # generate matrix representation of operator in a basis
            full_fock_basis = get_fock_basis(
                operator=operator, max_bose_occ=max_bose_occ
            )
            matrix = generate_matrix(operator, full_fock_basis)

        # Generate top left corner of overall circuit unitary
        upper_left_block = circuit.unitary(dtype=complex)[
            : 1 << system.number_of_system_qubits, : 1 << system.number_of_system_qubits
        ]
        unitary = upper_left_block * block_encoding_scaling_factor

    if return_numerics:
        print(NUMERICS)
    return circuit, unitary, matrix
