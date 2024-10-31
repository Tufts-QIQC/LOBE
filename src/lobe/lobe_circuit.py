from openparticle import *
import numpy as np
import cirq
from .system import System
from .block_encoding import add_lobe_oracle
from .usp import add_naive_usp
from .rescale import (
    bosonically_rescale_terms,
    rescale_terms_usp,
    get_number_of_active_bosonic_modes,
)
from .asp import get_target_state, add_prepare_circuit

from typing import Union, List


def lobe_circuit(
    operator: Union[ParticleOperator, List],
    max_bose_occ: int = 1,
    state_prep_protocol: str = "usp",
    return_unitary: bool = False,
):

    assert state_prep_protocol == "usp" or "asp"

    # prep gate counter:
    NUMERICS = {
        "left_elbows": 0,
        "right_elbows": 0,
        "rotations": 0,
        "ancillae_tracker": [0],
        "angles": [],
        "rescaling_factor": 1,
    }

    if isinstance(operator, List):
        terms = operator
        op_as_po = ParticleOperator({})
        for op in operator:
            op_as_po += op
        operator = op_as_po
    elif isinstance(operator, ParticleOperator):
        terms = operator.to_list()

    # Get target state for ASP
    rescaled_terms, NUMERICS["rescaling_factor"] = bosonically_rescale_terms(
        terms, max_bose_occ
    )

    number_of_modes = max([term.max_mode() for term in terms]) + 1

    number_of_ancillae = (
        1000  # Some arbitrary large number with most ancilla disregarded
    )
    number_of_index_qubits = max(int(np.ceil(np.log2(len(terms)))), 1)
    number_of_rotation_qubits = max(get_number_of_active_bosonic_modes(terms)) + 1

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
    if state_prep_protocol == "usp":
        rescaled_terms, usp_rescaling_factor = rescale_terms_usp(rescaled_terms)
        circuit += add_naive_usp(index_register)
        perform_coefficient_oracle = True
        NUMERICS["rescaling_factor"] *= (
            1 << number_of_index_qubits
        ) * usp_rescaling_factor
    elif state_prep_protocol == "asp":
        coefficients = [term.coeff for term in rescaled_terms]
        norm = sum(np.abs(coefficients))
        target_state = get_target_state(coefficients)
        circuit += add_prepare_circuit(
            index_register, target_state=target_state, numerics=NUMERICS
        )
        perform_coefficient_oracle = False
        NUMERICS["rescaling_factor"] *= norm
    circuit += add_lobe_oracle(
        rescaled_terms,
        validation,
        index_register,
        system,
        rotation_qubits,
        clean_ancillae,
        perform_coefficient_oracle=perform_coefficient_oracle,
        numerics=NUMERICS,
    )
    if state_prep_protocol == "usp":
        circuit += add_naive_usp(index_register)
    elif state_prep_protocol == "asp":
        circuit += add_prepare_circuit(
            index_register, target_state=target_state, numerics=NUMERICS
        )

    unitary = None
    matrix = None

    if return_unitary:
        # generate matrix representation of operator in a basis
        full_fock_basis = get_fock_basis(operator=operator, max_bose_occ=max_bose_occ)
        matrix = generate_matrix(operator, full_fock_basis)

        # Generate top left corner of overall circuit unitary
        upper_left_block = circuit.unitary(dtype=complex)[
            : 1 << system.number_of_system_qubits, : 1 << system.number_of_system_qubits
        ]
        unitary = upper_left_block * NUMERICS["rescaling_factor"]

    return circuit, NUMERICS, unitary, matrix
