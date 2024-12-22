from openparticle import ParticleOperator, generate_matrix
import numpy as np
import cirq
from src.lobe.system import System
from src.lobe._utils import get_basis_of_full_system
from src.lobe.fermionic import (
    fermionic_plus_hc_block_encoding,
    fermionic_product_block_encoding,
)
from src.lobe.bosonic import bosonic_mode_block_encoding
import pytest
import math


def _validate_block_encoding_bosonic(
    operator,
    active_indices,
    exponents,
    maximum_occupation_number,
    block_encoding_function,
):
    number_of_modes = max([term.max_mode for term in operator.to_list()]) + 1

    number_of_clean_ancillae = 100

    # Declare Qubits
    circuit = cirq.Circuit()
    control = cirq.LineQubit(0)
    block_encoding_ancilla = cirq.LineQubit(1)

    clean_ancillae = [cirq.LineQubit(i + 2) for i in range(number_of_clean_ancillae)]
    system = System(
        number_of_modes=number_of_modes,
        maximum_occupation_number=maximum_occupation_number,
        number_of_used_qubits=2 + number_of_clean_ancillae,
        has_fermions=False,
        has_bosons=True,
    )
    circuit.append(
        cirq.I.on_each(
            control,
            block_encoding_ancilla,
            *system.bosonic_system,
        )
    )

    # Flip control qubit so that we can focus on the 0-subspace of the control
    circuit.append(cirq.X.on(control))
    # Generate full Block-Encoding circuit
    gates, metrics = block_encoding_function(
        system,
        block_encoding_ancilla,
        active_indices,
        (exponents[0], exponents[1]),
        clean_ancillae=clean_ancillae,
        ctrls=([control], [1]),
    )
    circuit += gates
    # Flip control qubit so that we can focus on the 0-subspace of the control
    circuit.append(cirq.X.on(control))

    if len(circuit.all_qubits()) >= 20:
        pytest.skip(f"too many qubits to validate: {len(circuit.all_qubits())}")
    elif len(circuit.all_qubits()) >= 15:
        print("Testing singular quantum state")
        simulator = cirq.Simulator()
        random_system_state = 1j * np.random.uniform(
            -1, 1, 1 << system.number_of_system_qubits
        )
        random_system_state += np.random.uniform(
            -1, 1, 1 << system.number_of_system_qubits
        )
        random_system_state = random_system_state / np.linalg.norm(random_system_state)
        zero_state = np.zeros(
            1 << (len(circuit.all_qubits()) - system.number_of_system_qubits - 2),
            dtype=complex,
        )
        zero_state[0] = 1
        initial_control_state = [1, 0]
        initial_block_encoding_ancilla_state = [1, 0]
        initial_state = np.kron(
            np.kron(
                np.kron(initial_control_state, initial_block_encoding_ancilla_state),
                zero_state,
            ),
            random_system_state,
        )
        output_state = simulator.simulate(
            circuit, initial_state=initial_state
        ).final_state_vector
        final_state = output_state[: 1 << system.number_of_system_qubits]
        full_fock_basis = get_basis_of_full_system(
            number_of_modes,
            (1 << system.number_of_system_qubits),
            has_fermions=operator.has_fermions,
            has_antifermions=operator.has_antifermions,
            has_bosons=operator.has_bosons,
        )
        matrix = generate_matrix(operator, full_fock_basis)

        expected_final_state = matrix @ random_system_state
        expected_final_state = expected_final_state / np.linalg.norm(
            expected_final_state
        )
        assert np.allclose(
            expected_final_state, final_state / np.linalg.norm(final_state), atol=1e-6
        )
    else:
        full_fock_basis = get_basis_of_full_system(
            number_of_modes,
            maximum_occupation_number=maximum_occupation_number,
            has_fermions=operator.has_fermions,
            has_antifermions=operator.has_antifermions,
            has_bosons=operator.has_bosons,
        )
        matrix = generate_matrix(operator, full_fock_basis)

        expected_rescaling = (np.sqrt(maximum_occupation_number + 1)) ** (
            (sum(exponents))
        )
        upper_left_block = (
            expected_rescaling
            * circuit.unitary(dtype=complex)[
                : 1 << system.number_of_system_qubits,
                : 1 << system.number_of_system_qubits,
            ]
        )

        assert np.allclose(upper_left_block, matrix)


@pytest.mark.parametrize("number_of_modes", range(2, 4))
@pytest.mark.parametrize("active_mode", range(0, 4))
@pytest.mark.parametrize("Omega", [1, 3, 7])
@pytest.mark.parametrize("R", range(1, 4))
@pytest.mark.parametrize("S", range(1, 4))
def test_arbitrary_fermionic_product(number_of_modes, active_mode, Omega, R, S):
    if (Omega == 7) and (active_mode > 0):
        pytest.skip()

    active_mode = active_mode % number_of_modes
    operator = ParticleOperator(f"a{active_mode}^") ** R
    operator *= ParticleOperator(f"a{active_mode}") ** S
    exponents = (R, S)
    _validate_block_encoding_bosonic(
        operator,
        active_mode,
        exponents,
        Omega,
        bosonic_mode_block_encoding,
    )
