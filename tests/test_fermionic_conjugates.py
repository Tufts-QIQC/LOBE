from openparticle import ParticleOperator, generate_matrix
import numpy as np
import cirq
from src.lobe.system import System
from src.lobe._utils import get_basis_of_full_system
import pytest
import math


def _validate_block_encoding(operator, block_encoding_function):
    number_of_modes = max([term.max_mode() for term in operator.to_list()]) + 1

    number_of_clean_ancillae = 100

    # Declare Qubits
    circuit = cirq.Circuit()
    control = cirq.LineQubit(0)
    block_encoding_ancilla = cirq.LineQubit(1)

    clean_ancillae = [cirq.LineQubit(i + 2) for i in range(number_of_clean_ancillae)]
    system = System(
        number_of_modes=number_of_modes,
        maximum_occupation_number=1,
        number_of_used_qubits=2 + number_of_clean_ancillae,
        has_fermions=operator.has_fermions,
        has_antifermions=operator.has_antifermions,
        has_bosons=operator.has_bosons,
    )
    circuit.append(
        cirq.I.on_each(
            control,
            block_encoding_ancilla,
            *system.fermionic_register,
        )
    )

    if len(circuit.all_qubits()) >= 32:
        pytest.skip(f"too many qubits to build circuit: {len(circuit.all_qubits())}")

    # Flip control qubit so that we can focus on the 0-subspace of the control
    circuit.append(cirq.X.on(control))
    # Generate full Block-Encoding circuit
    circuit += block_encoding_function(
        operator, system, block_encoding_ancilla, clean_ancillae, ctrls=([control], [1])
    )
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
            1,
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
            1,
            has_fermions=operator.has_fermions,
            has_antifermions=operator.has_antifermions,
            has_bosons=operator.has_bosons,
        )
        matrix = generate_matrix(operator, full_fock_basis)

        upper_left_block = circuit.unitary(dtype=complex)[
            : 1 << system.number_of_system_qubits, : 1 << system.number_of_system_qubits
        ]

        assert np.allclose(upper_left_block, matrix)


def _validate_clean_ancillae_are_cleaned(operator, block_encoding_function):
    number_of_modes = max([term.max_mode() for term in operator.to_list()]) + 1

    number_of_clean_ancillae = 100

    # Declare Qubits
    circuit = cirq.Circuit()
    control = cirq.LineQubit(0)
    block_encoding_ancilla = cirq.LineQubit(1)

    clean_ancillae = [cirq.LineQubit(i + 2) for i in range(number_of_clean_ancillae)]
    system = System(
        number_of_modes=number_of_modes,
        maximum_occupation_number=1,
        number_of_used_qubits=2 + number_of_clean_ancillae,
        has_fermions=operator.has_fermions,
        has_antifermions=operator.has_antifermions,
        has_bosons=operator.has_bosons,
    )
    circuit.append(
        cirq.I.on_each(
            control,
            block_encoding_ancilla,
            *system.fermionic_register,
        )
    )

    if len(circuit.all_qubits()) >= 32:
        pytest.skip(f"too many qubits to build circuit: {len(circuit.all_qubits())}")

    # Flip control qubit so that we can focus on the 0-subspace of the control
    circuit.append(cirq.X.on(control))
    # Generate full Block-Encoding circuit
    circuit += block_encoding_function(
        operator, system, block_encoding_ancilla, clean_ancillae, ctrls=([control], [1])
    )
    # Flip control qubit so that we can focus on the 0-subspace of the control
    circuit.append(cirq.X.on(control))

    if len(circuit.all_qubits()) >= 20:
        pytest.skip(
            f"too many qubits to explicitly validate: {len(circuit.all_qubits())}"
        )
    else:
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
        initial_control_state = 1j * np.random.uniform(-1, 1, 2) + np.random.uniform(
            -1, 1, 2
        )
        initial_control_state = initial_control_state / np.linalg.norm(
            initial_control_state
        )
        initial_block_encoding_ancilla_state = 1j * np.random.uniform(-1, 1, 2)
        initial_block_encoding_ancilla_state += np.random.uniform(-1, 1, 2)
        initial_block_encoding_ancilla_state = (
            initial_block_encoding_ancilla_state
            / np.linalg.norm(initial_block_encoding_ancilla_state)
        )
        initial_state = np.kron(
            np.kron(
                np.kron(initial_control_state, initial_block_encoding_ancilla_state),
                zero_state,
            ),
            random_system_state,
        )
        final_state = simulator.simulate(
            circuit, initial_state=initial_state
        ).final_state_vector

        indices = final_state.nonzero()[0]
        for index in indices:
            bitstring = format(index, f"0{2+len(circuit.all_qubits())}b")[2:]
            for bit in bitstring[2 : -system.number_of_system_qubits]:
                if bit == "1":
                    assert False


def _validate_block_encoding_does_nothing_when_control_is_off(
    operator, block_encoding_function
):
    number_of_modes = max([term.max_mode() for term in operator.to_list()]) + 1

    number_of_clean_ancillae = 100

    # Declare Qubits
    circuit = cirq.Circuit()
    control = cirq.LineQubit(0)
    block_encoding_ancilla = cirq.LineQubit(1)

    clean_ancillae = [cirq.LineQubit(i + 2) for i in range(number_of_clean_ancillae)]
    system = System(
        number_of_modes=number_of_modes,
        maximum_occupation_number=1,
        number_of_used_qubits=2 + number_of_clean_ancillae,
        has_fermions=operator.has_fermions,
        has_antifermions=operator.has_antifermions,
        has_bosons=operator.has_bosons,
    )
    circuit.append(
        cirq.I.on_each(
            control,
            block_encoding_ancilla,
            *system.fermionic_register,
        )
    )

    if len(circuit.all_qubits()) >= 32:
        pytest.skip(f"too many qubits to build circuit: {len(circuit.all_qubits())}")

    # Generate full Block-Encoding circuit
    circuit += block_encoding_function(
        operator, system, block_encoding_ancilla, clean_ancillae, ctrls=([control], [1])
    )

    if len(circuit.all_qubits()) >= 20:
        pytest.skip(f"too many qubits to validate: {len(circuit.all_qubits())}")
    else:
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

        assert np.allclose(random_system_state, final_state, atol=1e-6)


MAX_MODES = 8
MAX_ACTIVE_MODES = 2
MIN_ACTIVE_MODES = 2


@pytest.mark.parametrize("trial", range(100))
def test_arbitrary_fermionic_operator_with_hc(trial):
    number_of_active_modes = np.random.random_integers(
        MIN_ACTIVE_MODES, MAX_ACTIVE_MODES
    )
    active_modes = np.random.choice(
        range(MAX_MODES + 1), size=number_of_active_modes, replace=False
    )
    daggers = np.random.choice([True, False], size=number_of_active_modes, replace=True)
    daggers = daggers[:number_of_active_modes]
    daggers = list(daggers)

    operator_string = ""
    for mode, dagger in zip(active_modes, daggers):
        operator_string += f" b{mode}"
        if dagger:
            operator_string += "^"

    conjugate_operator_string = ""
    for mode, dagger in zip(active_modes[::-1], daggers[::-1]):
        conjugate_operator_string += f" b{mode}"
        if not dagger:
            conjugate_operator_string += "^"

    operator = ParticleOperator(operator_string) + ParticleOperator(
        conjugate_operator_string
    )

    def arbitrary_fermionic_operator_with_hc_block_encoding(
        operator, system, block_encoding_ancilla, clean_ancillae=[], ctrls=([], [])
    ):
        assert len(ctrls[0]) == 1
        assert ctrls[1] == [1]
        gates = []
        active_modes = []
        dagger_values = [None] * len(system.fermionic_register)
        for op in list(operator.op_dict.keys())[0]:
            active_modes.append(op[1])
            dagger_values[op[1]] = op[2]

        temporary_computations = []
        parity_qubits = clean_ancillae[: len(active_modes) - 1]
        for i, active_mode in enumerate(active_modes[:-1]):
            parity_qubit = parity_qubits[i]
            temporary_computations.append(
                cirq.Moment(
                    cirq.X.on(parity_qubit).controlled_by(
                        system.fermionic_register[active_mode],
                        control_values=[not dagger_values[active_mode]],
                    )
                )
            )
            temporary_computations.append(
                cirq.Moment(
                    cirq.X.on(parity_qubit).controlled_by(
                        system.fermionic_register[active_modes[i + 1]],
                        control_values=[not dagger_values[active_modes[i + 1]]],
                    )
                )
            )

        # Use left-elbow to store temporary logical AND of parity qubits and control
        temporary_qbool = clean_ancillae[len(active_modes) - 1]
        temporary_computations.append(
            cirq.Moment(
                cirq.X.on(temporary_qbool).controlled_by(
                    *parity_qubits,
                    *ctrls[0],
                    control_values=([0] * len(parity_qubits)) + ctrls[1],
                )
            )
        )
        gates += temporary_computations

        # Flip block-encoding ancilla
        gates.append(
            cirq.Moment(
                cirq.X.on(block_encoding_ancilla).controlled_by(
                    temporary_qbool, control_values=[0]
                )
            )
        )

        # Reset clean ancillae
        gates += temporary_computations[::-1]

        gates.append(
            cirq.Moment(
                cirq.X.on(block_encoding_ancilla).controlled_by(
                    *ctrls[0], control_values=[0]
                )
            )
        )

        # Update system
        active_qubits = [
            system.fermionic_register[active_mode] for active_mode in active_modes
        ]
        number_of_swaps = math.comb(number_of_active_modes, 2)
        if number_of_swaps % 2:
            gates.append(
                cirq.Moment(
                    cirq.Z.on(ctrls[0][0]).controlled_by(
                        active_qubits[0],
                        control_values=[dagger_values[active_modes[0]]],
                    )
                )
            )

        for active_mode in active_modes[::-1]:
            for system_qubit in system.fermionic_register[:active_mode]:
                gates.append(
                    cirq.Moment(
                        cirq.Z.on(system_qubit).controlled_by(
                            *ctrls[0], control_values=ctrls[1]
                        )
                    )
                )
            gates.append(
                cirq.Moment(
                    cirq.X.on(system.fermionic_register[active_mode]).controlled_by(
                        *ctrls[0], control_values=ctrls[1]
                    )
                )
            )

        return gates

    _validate_clean_ancillae_are_cleaned(
        operator, arbitrary_fermionic_operator_with_hc_block_encoding
    )
    _validate_block_encoding(
        operator, arbitrary_fermionic_operator_with_hc_block_encoding
    )
    _validate_block_encoding_does_nothing_when_control_is_off(
        operator, arbitrary_fermionic_operator_with_hc_block_encoding
    )
