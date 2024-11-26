from openparticle import ParticleOperator, generate_matrix
import numpy as np
import cirq
from src.lobe.system import System
from src.lobe._utils import get_basis_of_full_system
import pytest


def _test_helper(operator, block_encoding_function):
    number_of_modes = max([term.max_mode() for term in operator.to_list()]) + 1

    number_of_clean_ancillae = 100

    # Declare Qubits
    circuit = cirq.Circuit()
    block_encoding_ancilla = cirq.LineQubit(0)

    clean_ancillae = [cirq.LineQubit(i + 1) for i in range(number_of_clean_ancillae)]
    system = System(
        number_of_modes=number_of_modes,
        maximum_occupation_number=1,
        number_of_used_qubits=1 + number_of_clean_ancillae,
        has_fermions=operator.has_fermions,
        has_antifermions=operator.has_antifermions,
        has_bosons=operator.has_bosons,
    )
    circuit.append(
        cirq.I.on_each(
            block_encoding_ancilla,
            *system.fermionic_register,
        )
    )

    if len(circuit.all_qubits()) >= 32:
        pytest.skip(f"too many qubits {len(circuit.all_qubits())} to build circuit")

    # Generate full Block-Encoding circuit
    circuit += block_encoding_function(
        operator,
        system,
        block_encoding_ancilla,
        clean_ancillae,
    )

    if len(circuit.all_qubits()) >= 14:
        pytest.skip(
            f"too many qubits {len(circuit.all_qubits())} to explicitly validate"
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


@pytest.mark.parametrize("number_of_modes", range(2, 8))
@pytest.mark.parametrize("active_mode", range(0, 7))
def test_single_fermionic_operator(number_of_modes, active_mode):
    active_mode = active_mode % number_of_modes
    operator = ParticleOperator(f"b{active_mode}^")
    operator += ParticleOperator(f"b{active_mode}")

    def single_fermionic_operator_block_encoding(
        operator, system, block_encoding_ancilla, clean_ancillae=[], ctrls=([], [])
    ):
        gates = []
        active_mode = list(operator.op_dict.keys())[0][0][1]
        validate_operator = ParticleOperator(f"b{active_mode}^")
        validate_operator += ParticleOperator(f"b{active_mode}")
        # assert operator == validate_operator
        for system_qubit in system.fermionic_register[:active_mode]:
            gates.append(
                cirq.Z.on(system_qubit).controlled_by(
                    *ctrls[0], control_values=ctrls[1]
                )
            )
        gates.append(
            cirq.X.on(system.fermionic_register[active_mode]).controlled_by(
                *ctrls[0], control_values=ctrls[1]
            )
        )
        return gates

    _test_helper(operator, single_fermionic_operator_block_encoding)


@pytest.mark.parametrize("number_of_modes", range(2, 8))
@pytest.mark.parametrize("active_mode", range(0, 7))
def test_fermionic_number_operator(number_of_modes, active_mode):
    active_mode = active_mode % number_of_modes
    operator = ParticleOperator(f"b{active_mode}^ b{active_mode}")

    def fermionic_number_operator_block_encoding(
        operator, system, block_encoding_ancilla, clean_ancillae=[], ctrls=([], [])
    ):
        gates = []
        active_mode = list(operator.op_dict.keys())[0][0][1]
        validate_operator = ParticleOperator(f"b{active_mode}^ b{active_mode}")
        # assert operator == validate_operator
        gates.append(
            cirq.X.on(block_encoding_ancilla).controlled_by(
                system.fermionic_register[active_mode],
                *ctrls[0],
                control_values=[0] + ctrls[1],
            )
        )
        return gates

    _test_helper(operator, fermionic_number_operator_block_encoding)


@pytest.mark.parametrize("number_of_modes", range(2, 8))
@pytest.mark.parametrize("first_active_mode", range(0, 7))
@pytest.mark.parametrize("second_active_mode", range(0, 7))
def test_two_site_fermionic_operator(
    number_of_modes, first_active_mode, second_active_mode
):
    first_active_mode = first_active_mode % number_of_modes
    second_active_mode = second_active_mode % number_of_modes

    if first_active_mode == second_active_mode:
        pytest.skip()

    operator = ParticleOperator(f"b{first_active_mode}^ b{second_active_mode}")
    operator += ParticleOperator(f"b{second_active_mode}^ b{first_active_mode}")

    def two_site_fermionic_operator_block_encoding(
        operator, system, block_encoding_ancilla, clean_ancillae=[], ctrls=([], [])
    ):
        gates = []
        first_active_mode = list(operator.op_dict.keys())[0][0][1]
        second_active_mode = list(operator.op_dict.keys())[0][1][1]
        validate_operator = ParticleOperator(
            f"b{first_active_mode}^ b{second_active_mode}"
        )
        validate_operator += ParticleOperator(
            f"b{second_active_mode}^ b{first_active_mode}"
        )
        # assert operator == validate_operator

        # Compute parity of mode occupation using clean ancilla
        parity_qubit = clean_ancillae[0]
        gates.append(
            cirq.X.on(parity_qubit).controlled_by(
                system.fermionic_register[first_active_mode]
            )
        )
        gates.append(
            cirq.X.on(parity_qubit).controlled_by(
                system.fermionic_register[second_active_mode]
            )
        )

        # Use left-elbow to store temporary logical AND of parity qubit and control
        temporary_qbool = clean_ancillae[1]
        gates.append(
            cirq.X.on(temporary_qbool).controlled_by(
                parity_qubit, *ctrls[0], control_values=[1] + ctrls[1]
            )
        )

        # Update system
        for system_qubit in system.fermionic_register[
            min(first_active_mode, second_active_mode)
            + 1 : max(first_active_mode, second_active_mode)
        ]:
            gates.append(
                cirq.Z.on(system_qubit).controlled_by(
                    temporary_qbool, control_values=[1]
                )
            )
        gates.append(
            cirq.X.on(system.fermionic_register[first_active_mode]).controlled_by(
                temporary_qbool, control_values=[1]
            )
        )
        gates.append(
            cirq.X.on(system.fermionic_register[second_active_mode]).controlled_by(
                temporary_qbool, control_values=[1]
            )
        )

        # Flip block-encoding ancilla
        gates.append(
            cirq.X.on(block_encoding_ancilla).controlled_by(
                temporary_qbool, control_values=[0]
            )
        )

        # Reset clean ancillae
        gates.append(
            cirq.X.on(temporary_qbool).controlled_by(
                parity_qubit, *ctrls[0], control_values=[1] + ctrls[1]
            )
        )
        gates.append(
            cirq.X.on(parity_qubit).controlled_by(
                system.fermionic_register[second_active_mode]
            )
        )
        gates.append(
            cirq.X.on(parity_qubit).controlled_by(
                system.fermionic_register[first_active_mode]
            )
        )

        return gates

    _test_helper(operator, two_site_fermionic_operator_block_encoding)


@pytest.mark.parametrize("number_of_modes", range(2, 8))
@pytest.mark.parametrize("first_active_mode", range(0, 7))
@pytest.mark.parametrize("second_active_mode", range(0, 7))
@pytest.mark.parametrize("third_active_mode", range(0, 7))
def test_three_site_all_same_type_fermionic_operator(
    number_of_modes, first_active_mode, second_active_mode, third_active_mode
):
    first_active_mode = first_active_mode % number_of_modes
    second_active_mode = second_active_mode % number_of_modes
    third_active_mode = third_active_mode % number_of_modes

    if (
        (first_active_mode == second_active_mode)
        or (first_active_mode == third_active_mode)
        or (second_active_mode == third_active_mode)
    ):
        pytest.skip()

    operator = ParticleOperator(
        f"b{first_active_mode} b{second_active_mode} b{third_active_mode}"
    )
    operator += ParticleOperator(
        f"b{third_active_mode}^ b{second_active_mode}^ b{first_active_mode}^"
    )

    def three_site_fermionic_operator_all_same_type_block_encoding(
        operator, system, block_encoding_ancilla, clean_ancillae=[], ctrls=([], [])
    ):
        gates = []
        first_active_mode = list(operator.op_dict.keys())[0][0][1]
        second_active_mode = list(operator.op_dict.keys())[0][1][1]
        third_active_mode = list(operator.op_dict.keys())[0][2][1]
        validate_operator = ParticleOperator(
            f"b{first_active_mode} b{second_active_mode} b{third_active_mode}"
        )
        validate_operator += ParticleOperator(
            f"b{third_active_mode}^ b{second_active_mode}^ b{first_active_mode}^"
        )
        # assert operator == validate_operator

        # Compute parity of first and second mode occupation using clean ancilla
        first_parity_qubit = clean_ancillae[0]
        gates.append(
            cirq.X.on(first_parity_qubit).controlled_by(
                system.fermionic_register[first_active_mode]
            )
        )
        gates.append(
            cirq.X.on(first_parity_qubit).controlled_by(
                system.fermionic_register[second_active_mode]
            )
        )

        # Compute parity of second and third mode occupation using clean ancilla
        second_parity_qubit = clean_ancillae[1]
        gates.append(
            cirq.X.on(second_parity_qubit).controlled_by(
                system.fermionic_register[second_active_mode]
            )
        )
        gates.append(
            cirq.X.on(second_parity_qubit).controlled_by(
                system.fermionic_register[third_active_mode]
            )
        )

        # Use left-elbow to store temporary logical AND of parity qubits and control
        temporary_qbool = clean_ancillae[2]
        gates.append(
            cirq.X.on(temporary_qbool).controlled_by(
                first_parity_qubit,
                second_parity_qubit,
                *ctrls[0],
                control_values=[0] + [0] + ctrls[1],
            )
        )

        # Update system
        for system_qubit in system.fermionic_register[:third_active_mode]:
            gates.append(
                cirq.Z.on(system_qubit).controlled_by(
                    temporary_qbool, control_values=[1]
                )
            )
        gates.append(
            cirq.X.on(system.fermionic_register[third_active_mode]).controlled_by(
                temporary_qbool, control_values=[1]
            )
        )
        for system_qubit in system.fermionic_register[:second_active_mode]:
            gates.append(
                cirq.Z.on(system_qubit).controlled_by(
                    temporary_qbool, control_values=[1]
                )
            )
        gates.append(
            cirq.X.on(system.fermionic_register[second_active_mode]).controlled_by(
                temporary_qbool, control_values=[1]
            )
        )
        for system_qubit in system.fermionic_register[:first_active_mode]:
            gates.append(
                cirq.Z.on(system_qubit).controlled_by(
                    temporary_qbool, control_values=[1]
                )
            )
        gates.append(
            cirq.X.on(system.fermionic_register[first_active_mode]).controlled_by(
                temporary_qbool, control_values=[1]
            )
        )
        gates.append(
            cirq.Z.on(system.fermionic_register[second_active_mode]).controlled_by(
                temporary_qbool, control_values=[1]
            )
        )

        # Flip block-encoding ancilla
        gates.append(
            cirq.X.on(block_encoding_ancilla).controlled_by(
                temporary_qbool, control_values=[0]
            )
        )

        # Reset clean ancillae
        gates.append(
            cirq.X.on(temporary_qbool).controlled_by(
                first_parity_qubit,
                second_parity_qubit,
                *ctrls[0],
                control_values=[0] + [0] + ctrls[1],
            )
        )
        gates.append(
            cirq.X.on(second_parity_qubit).controlled_by(
                system.fermionic_register[third_active_mode]
            )
        )
        gates.append(
            cirq.X.on(second_parity_qubit).controlled_by(
                system.fermionic_register[second_active_mode]
            )
        )
        gates.append(
            cirq.X.on(first_parity_qubit).controlled_by(
                system.fermionic_register[second_active_mode]
            )
        )
        gates.append(
            cirq.X.on(first_parity_qubit).controlled_by(
                system.fermionic_register[first_active_mode]
            )
        )

        return gates

    _test_helper(operator, three_site_fermionic_operator_all_same_type_block_encoding)


@pytest.mark.parametrize("number_of_modes", range(2, 8))
@pytest.mark.parametrize("first_active_mode", range(0, 7))
@pytest.mark.parametrize("second_active_mode", range(0, 7))
@pytest.mark.parametrize("third_active_mode", range(0, 7))
def test_three_site_different_types_fermionic_operator(
    number_of_modes, first_active_mode, second_active_mode, third_active_mode
):
    first_active_mode = first_active_mode % number_of_modes
    second_active_mode = second_active_mode % number_of_modes
    third_active_mode = third_active_mode % number_of_modes

    if (
        (first_active_mode == second_active_mode)
        or (first_active_mode == third_active_mode)
        or (second_active_mode == third_active_mode)
    ):
        pytest.skip()

    def three_site_fermionic_operator_different_types_block_encoding(
        operator, system, block_encoding_ancilla, clean_ancillae=[], ctrls=([], [])
    ):
        # NOTE: Assumes that the first term in the operator list is the one with one creation op and two annihilation ops
        gates = []
        first_active_mode = list(operator.op_dict.keys())[0][0][1]
        second_active_mode = list(operator.op_dict.keys())[0][1][1]
        third_active_mode = list(operator.op_dict.keys())[0][2][1]
        validate_operator = ParticleOperator(
            f"b{first_active_mode} b{second_active_mode} b{third_active_mode}"
        )
        validate_operator += ParticleOperator(
            f"b{third_active_mode}^ b{second_active_mode}^ b{first_active_mode}^"
        )
        # assert operator == validate_operator

        # Compute parity of first and second mode occupation using clean ancilla
        first_parity_qubit = clean_ancillae[0]
        gates.append(
            cirq.X.on(first_parity_qubit).controlled_by(
                system.fermionic_register[first_active_mode]
            )
        )
        gates.append(
            cirq.X.on(first_parity_qubit).controlled_by(
                system.fermionic_register[second_active_mode]
            )
        )

        # Compute parity of second and third mode occupation using clean ancilla
        second_parity_qubit = clean_ancillae[1]
        gates.append(
            cirq.X.on(second_parity_qubit).controlled_by(
                system.fermionic_register[second_active_mode]
            )
        )
        gates.append(
            cirq.X.on(second_parity_qubit).controlled_by(
                system.fermionic_register[third_active_mode]
            )
        )

        # Use left-elbow to store temporary logical AND of parity qubits and control
        temporary_qbool = clean_ancillae[2]
        gates.append(
            cirq.X.on(temporary_qbool).controlled_by(
                first_parity_qubit,
                second_parity_qubit,
                *ctrls[0],
                control_values=[1] + [0] + ctrls[1],
            )
        )

        # Update system
        for system_qubit in system.fermionic_register[:third_active_mode]:
            gates.append(
                cirq.Z.on(system_qubit).controlled_by(
                    temporary_qbool, control_values=[1]
                )
            )
        gates.append(
            cirq.X.on(system.fermionic_register[third_active_mode]).controlled_by(
                temporary_qbool, control_values=[1]
            )
        )
        for system_qubit in system.fermionic_register[:second_active_mode]:
            gates.append(
                cirq.Z.on(system_qubit).controlled_by(
                    temporary_qbool, control_values=[1]
                )
            )
        gates.append(
            cirq.X.on(system.fermionic_register[second_active_mode]).controlled_by(
                temporary_qbool, control_values=[1]
            )
        )
        for system_qubit in system.fermionic_register[:first_active_mode]:
            gates.append(
                cirq.Z.on(system_qubit).controlled_by(
                    temporary_qbool, control_values=[1]
                )
            )
        gates.append(
            cirq.X.on(system.fermionic_register[first_active_mode]).controlled_by(
                temporary_qbool, control_values=[1]
            )
        )
        gates.append(
            cirq.Z.on(system.fermionic_register[second_active_mode]).controlled_by(
                temporary_qbool, control_values=[1]
            )
        )

        # Flip block-encoding ancilla
        gates.append(
            cirq.X.on(block_encoding_ancilla).controlled_by(
                temporary_qbool, control_values=[0]
            )
        )

        # Reset clean ancillae
        gates.append(
            cirq.X.on(temporary_qbool).controlled_by(
                first_parity_qubit,
                second_parity_qubit,
                *ctrls[0],
                control_values=[1] + [0] + ctrls[1],
            )
        )
        gates.append(
            cirq.X.on(second_parity_qubit).controlled_by(
                system.fermionic_register[third_active_mode]
            )
        )
        gates.append(
            cirq.X.on(second_parity_qubit).controlled_by(
                system.fermionic_register[second_active_mode]
            )
        )
        gates.append(
            cirq.X.on(first_parity_qubit).controlled_by(
                system.fermionic_register[second_active_mode]
            )
        )
        gates.append(
            cirq.X.on(first_parity_qubit).controlled_by(
                system.fermionic_register[first_active_mode]
            )
        )

        return gates

    operator = ParticleOperator(
        f"b{first_active_mode}^ b{second_active_mode} b{third_active_mode}"
    )
    operator += ParticleOperator(
        f"b{third_active_mode}^ b{second_active_mode}^ b{first_active_mode}"
    )
    _test_helper(operator, three_site_fermionic_operator_different_types_block_encoding)
