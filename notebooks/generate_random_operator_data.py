import sys, os

sys.path.append(os.path.join(os.path.dirname(os.path.realpath("__file__")), "../src"))
from openparticle import ParticleOperator
import numpy as np
import cirq
import json
from lobe.rescale import (
    bosonically_rescale_terms,
    rescale_terms_usp,
    get_number_of_active_bosonic_modes,
)
from lobe.block_encoding import add_lobe_oracle
from lobe.asp import add_prepare_circuit, get_target_state
from lobe.usp import add_naive_usp
from lobe.system import System
from time import time

# VALUES_OF_L = np.linspace(1, 10000, 20)
# VALUES_OF_I = [1, 15, 3, 13, 5, 11, 7, 9]
# VALUES_OF_OMEGA = [1, 3, 7, 15]
# LENGTHS_OF_TERMS = [1, 2, 3, 4, 5]

VALUES_OF_L = np.linspace(1, 10000, 20)
VALUES_OF_I = [15]
VALUES_OF_OMEGA = [31, 63, 127, 255]
LENGTHS_OF_TERMS = [5]

COUNTER = 0

for number_of_modes in VALUES_OF_I:
    for maximum_bosonic_occupation in VALUES_OF_OMEGA:
        for maximum_length_of_terms in LENGTHS_OF_TERMS:

            with open("random_operators_numerical_data.csv", "a") as f:
                for number_of_terms in VALUES_OF_L:
                    number_of_terms = int(number_of_terms)

                    print("---- Starting Trial {}----".format(COUNTER))
                    print(
                        "L: {}\nI: {}\n(B+A): {}\nOmega: {}".format(
                            number_of_terms,
                            number_of_modes,
                            maximum_length_of_terms,
                            maximum_bosonic_occupation,
                        )
                    )
                    start = time()

                    operator = ParticleOperator.random(
                        particle_types=["fermion", "antifermion", "boson"],
                        n_terms=number_of_terms,
                        max_mode=number_of_modes,
                        max_len_of_terms=maximum_length_of_terms,
                        complex_coeffs=False,
                    )
                    operator.remove_identity()
                    terms = operator.to_list()
                    while (
                        max([term.max_mode() for term in terms]) + 1 != number_of_modes
                    ):
                        operator = ParticleOperator.random(
                            particle_types=["fermion", "antifermion", "boson"],
                            n_terms=number_of_terms,
                            max_mode=number_of_modes,
                            max_len_of_terms=maximum_length_of_terms,
                            complex_coeffs=False,
                        )
                        operator.remove_identity()
                        terms = operator.to_list()

                    print(
                        "Time to Generate Operator: {} (s)".format(
                            round(time() - start, 2)
                        )
                    )

                    bosonically_rescaled_terms, bosonic_rescaling_factor = (
                        bosonically_rescale_terms(terms, maximum_bosonic_occupation)
                    )
                    coefficients = [term.coeff for term in bosonically_rescaled_terms]

                    norm = sum(np.abs(coefficients))
                    target_state = get_target_state(coefficients)
                    asp_rescaling_factor = bosonic_rescaling_factor * norm

                    number_of_ancillae = 1000  # Some arbitrary large number with most ancilla disregarded
                    number_of_index_qubits = max(int(np.ceil(np.log2(len(terms)))), 1)
                    number_of_rotation_qubits = (
                        max(get_number_of_active_bosonic_modes(terms)) + 1
                    )

                    usp_rescaled_terms, usp_rescaling_factor = rescale_terms_usp(
                        bosonically_rescaled_terms
                    )
                    usp_rescaling_factor *= bosonic_rescaling_factor * (
                        1 << number_of_index_qubits
                    )

                    # Declare Qubits
                    validation = cirq.LineQubit(0)
                    clean_ancillae = [
                        cirq.LineQubit(i + 1) for i in range(number_of_ancillae)
                    ]
                    rotation_qubits = [
                        cirq.LineQubit(i + 1 + number_of_ancillae)
                        for i in range(number_of_rotation_qubits)
                    ]
                    index_register = [
                        cirq.LineQubit(
                            i + 1 + number_of_ancillae + number_of_rotation_qubits
                        )
                        for i in range(number_of_index_qubits)
                    ]
                    system = System(
                        number_of_modes=number_of_modes,
                        maximum_occupation_number=maximum_bosonic_occupation,
                        number_of_used_qubits=1
                        + number_of_ancillae
                        + number_of_rotation_qubits
                        + number_of_index_qubits,
                        has_fermions=operator.has_fermions,
                        has_antifermions=operator.has_antifermions,
                        has_bosons=operator.has_bosons,
                    )

                    #### USP Circuit Generation
                    USP_numerics = {
                        "left_elbows": 0,
                        "right_elbows": 0,
                        "rotations": 0,
                        "ancillae_tracker": [
                            1 + number_of_rotation_qubits + number_of_index_qubits
                        ],
                        "angles": [],
                        "number_of_nonclifford_rotations": 0,
                        "rescaling_factor": usp_rescaling_factor,
                        "number_of_modes": number_of_modes,
                        "maximum_bosonic_occupation": maximum_bosonic_occupation,
                        "maximum_number_of_active_bosonic_modes": number_of_rotation_qubits
                        - 1,
                        "number_of_terms": len(terms),
                    }
                    circuit = cirq.Circuit()
                    circuit.append(cirq.I.on_each(*system.fermionic_register))
                    circuit.append(cirq.I.on_each(*system.antifermionic_register))
                    for bosonic_reg in system.bosonic_system:
                        circuit.append(cirq.I.on_each(*bosonic_reg))
                    circuit.append(cirq.X.on(validation))
                    circuit += add_naive_usp(index_register)
                    circuit += add_lobe_oracle(
                        usp_rescaled_terms,
                        validation,
                        index_register,
                        system,
                        rotation_qubits,
                        clean_ancillae,
                        perform_coefficient_oracle=True,
                        decompose=True,
                        numerics=USP_numerics,
                    )
                    circuit += add_naive_usp(index_register)
                    USP_numerics["number_of_ancillae"] = max(
                        USP_numerics["ancillae_tracker"]
                    )
                    USP_numerics["number_of_qubits"] = (
                        max(USP_numerics["ancillae_tracker"])
                        + system.number_of_system_qubits
                    )
                    for angle in USP_numerics["angles"]:
                        angle = np.abs(angle)
                        if not np.isclose(angle % np.pi / 4, 0):
                            USP_numerics["number_of_nonclifford_rotations"] += 1

                    f.write(
                        "\n{},{},{},{},{},{},{},{},{},{}".format(
                            number_of_terms,
                            number_of_modes,
                            maximum_length_of_terms,
                            maximum_bosonic_occupation,
                            "USP",
                            USP_numerics["number_of_qubits"],
                            USP_numerics["number_of_nonclifford_rotations"],
                            USP_numerics["left_elbows"],
                            USP_numerics["right_elbows"],
                            usp_rescaling_factor,
                        )
                    )

                    #### ASP Circuit Generation
                    ASP_numerics = {
                        "left_elbows": 0,
                        "right_elbows": 0,
                        "rotations": 0,
                        "ancillae_tracker": [
                            1 + number_of_rotation_qubits - 1 + number_of_index_qubits
                        ],
                        "angles": [],
                        "number_of_nonclifford_rotations": 0,
                        "rescaling_factor": asp_rescaling_factor,
                        "number_of_modes": number_of_modes,
                        "maximum_bosonic_occupation": maximum_bosonic_occupation,
                        "maximum_number_of_active_bosonic_modes": number_of_rotation_qubits
                        - 1,
                        "number_of_terms": len(terms),
                    }
                    circuit = cirq.Circuit()
                    circuit.append(cirq.I.on_each(*system.fermionic_register))
                    circuit.append(cirq.I.on_each(*system.antifermionic_register))
                    for bosonic_reg in system.bosonic_system:
                        circuit.append(cirq.I.on_each(*bosonic_reg))
                    circuit.append(cirq.X.on(validation))
                    circuit += add_prepare_circuit(
                        index_register,
                        target_state=target_state,
                        numerics=ASP_numerics,
                        clean_ancillae=clean_ancillae,
                    )
                    circuit += add_lobe_oracle(
                        bosonically_rescaled_terms,
                        validation,
                        index_register,
                        system,
                        rotation_qubits,
                        clean_ancillae,
                        perform_coefficient_oracle=False,
                        decompose=True,
                        numerics=ASP_numerics,
                    )
                    circuit += add_prepare_circuit(
                        index_register,
                        target_state=target_state,
                        dagger=True,
                        numerics=ASP_numerics,
                        clean_ancillae=clean_ancillae,
                    )
                    ASP_numerics["number_of_ancillae"] = max(
                        ASP_numerics["ancillae_tracker"]
                    )
                    ASP_numerics["number_of_qubits"] = (
                        max(ASP_numerics["ancillae_tracker"])
                        + system.number_of_system_qubits
                    )
                    for angle in ASP_numerics["angles"]:
                        angle = np.abs(angle)
                        if not np.isclose(angle % np.pi / 4, 0):
                            ASP_numerics["number_of_nonclifford_rotations"] += 1

                    f.write(
                        "\n{},{},{},{},{},{},{},{},{},{}".format(
                            number_of_terms,
                            number_of_modes,
                            maximum_length_of_terms,
                            maximum_bosonic_occupation,
                            "ASP",
                            ASP_numerics["number_of_qubits"],
                            ASP_numerics["number_of_nonclifford_rotations"],
                            ASP_numerics["left_elbows"],
                            ASP_numerics["right_elbows"],
                            asp_rescaling_factor,
                        )
                    )
                    COUNTER += 1
                    print("Time Elapsed: {} (s)".format(round(time() - start, 2)))
                f.close()
