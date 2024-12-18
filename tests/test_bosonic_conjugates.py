from openparticle import ParticleOperator, generate_matrix
import numpy as np
import cirq
from src.lobe.system import System
from src.lobe._utils import get_basis_of_full_system
from src.lobe.addition import add_classical_value_gate_efficient
from src.lobe.block_encoding import _add_bosonic_rotations
import pytest


def _test_helper(
    operator,
    maximum_occupation_number,
    expected_rescaling_factor,
    block_encoding_function,
):
    number_of_modes = max([term.max_mode() for term in operator.to_list()]) + 1

    number_of_clean_ancillae = 100

    # Declare Qubits
    circuit = cirq.Circuit()
    block_encoding_ancillae = [cirq.LineQubit(0), cirq.LineQubit(1)]

    clean_ancillae = [cirq.LineQubit(i + 2) for i in range(number_of_clean_ancillae)]
    system = System(
        number_of_modes=number_of_modes,
        maximum_occupation_number=maximum_occupation_number,
        number_of_used_qubits=2 + number_of_clean_ancillae,
        has_fermions=operator.has_fermions,
        has_antifermions=operator.has_antifermions,
        has_bosons=operator.has_bosons,
    )
    circuit.append(
        cirq.I.on_each(
            block_encoding_ancillae,
        )
    )
    for register in system.bosonic_system:
        circuit.append(cirq.I.on_each(*register))

    if len(circuit.all_qubits()) >= 32:
        pytest.skip(f"too many qubits {len(circuit.all_qubits())} to build circuit")

    # Generate full Block-Encoding circuit
    circuit += block_encoding_function(
        operator,
        system,
        block_encoding_ancillae,
        clean_ancillae,
    )

    if len(circuit.all_qubits()) >= 14:
        pytest.skip(
            f"too many qubits {len(circuit.all_qubits())} to explicitly validate"
        )
    else:
        full_fock_basis = get_basis_of_full_system(
            number_of_modes,
            maximum_occupation_number,
            has_fermions=operator.has_fermions,
            has_antifermions=operator.has_antifermions,
            has_bosons=operator.has_bosons,
        )
        matrix = generate_matrix(operator, full_fock_basis)

        upper_left_block = circuit.unitary(dtype=complex)[
            : 1 << system.number_of_system_qubits, : 1 << system.number_of_system_qubits
        ]
        rescaled_block = upper_left_block * expected_rescaling_factor

        assert np.allclose(rescaled_block, matrix)


@pytest.mark.parametrize("number_of_modes", range(2, 4))
@pytest.mark.parametrize("active_mode", range(0, 4))
@pytest.mark.parametrize("maximum_occupation_number", [1, 3, 7])
@pytest.mark.parametrize("exponent", range(1, 8))
def test_single_bosonic_operator(
    number_of_modes, active_mode, maximum_occupation_number, exponent
):
    if (maximum_occupation_number == 7) and (active_mode > 0):
        pytest.skip()
    active_mode = active_mode % number_of_modes
    operator = ParticleOperator(f"a{active_mode}^") ** exponent
    operator += ParticleOperator(f"a{active_mode}") ** exponent
    expected_rescaling_factor = 2 * (np.sqrt(maximum_occupation_number + 1)) ** exponent

    def single_bosonic_operator_block_encoding(
        operator, system, block_encoding_ancillae, clean_ancillae=[], ctrls=([], [])
    ):
        gates = []
        active_mode = list(operator.op_dict.keys())[0][0][1]
        validate_operator = ParticleOperator(f"a{active_mode}^")
        validate_operator += ParticleOperator(f"a{active_mode}")
        # assert operator == validate_operator

        gates.append(cirq.H.on(block_encoding_ancillae[0]))

        adder_controls = (ctrls[0] + [block_encoding_ancillae[0]], ctrls[1] + [0])
        gates += add_classical_value_gate_efficient(
            system.bosonic_system[active_mode],
            exponent,
            clean_ancillae=clean_ancillae,
            ctrls=adder_controls,
        )

        gates += _add_bosonic_rotations(
            block_encoding_ancillae[1],
            system.bosonic_system[active_mode],
            0,
            exponent,
            clean_ancillae=clean_ancillae,
            ctrls=ctrls,
        )

        adder_controls = (ctrls[0] + [block_encoding_ancillae[0]], ctrls[1] + [1])
        gates += add_classical_value_gate_efficient(
            system.bosonic_system[active_mode],
            -exponent,
            clean_ancillae=clean_ancillae,
            ctrls=adder_controls,
        )

        gates.append(cirq.H.on(block_encoding_ancillae[0]))
        return gates

    _test_helper(
        operator,
        maximum_occupation_number,
        expected_rescaling_factor,
        single_bosonic_operator_block_encoding,
    )


@pytest.mark.parametrize("number_of_modes", range(2, 4))
@pytest.mark.parametrize("active_mode", range(0, 4))
@pytest.mark.parametrize("maximum_occupation_number", [1, 3, 7])
@pytest.mark.parametrize("R", range(1, 8))
@pytest.mark.parametrize("S", range(1, 8))
def test_two_bosonic_operators(
    number_of_modes, active_mode, maximum_occupation_number, R, S
):
    if (maximum_occupation_number == 7) and (active_mode > 0):
        pytest.skip()
    if R == S:
        pytest.skip()
    active_mode = active_mode % number_of_modes
    operator = (ParticleOperator(f"a{active_mode}^") ** R) * (
        ParticleOperator(f"a{active_mode}") ** S
    )
    operator += (ParticleOperator(f"a{active_mode}^") ** S) * (
        ParticleOperator(f"a{active_mode}") ** R
    )
    expected_rescaling_factor = 2 * (
        (np.sqrt(maximum_occupation_number + 1)) ** ((R + S))
    )

    def two_bosonic_operators_block_encoding(
        operator, system, block_encoding_ancillae, clean_ancillae=[], ctrls=([], [])
    ):
        gates = []
        active_mode = list(operator.op_dict.keys())[0][0][1]

        gates.append(cirq.H.on(block_encoding_ancillae[0]))

        adder_controls = (ctrls[0] + [block_encoding_ancillae[0]], ctrls[1] + [0])
        gates += add_classical_value_gate_efficient(
            system.bosonic_system[active_mode],
            R - S,
            clean_ancillae=clean_ancillae,
            ctrls=adder_controls,
        )

        gates += _add_multi_bosonic_rotations(
            block_encoding_ancillae[1],
            system.bosonic_system[active_mode],
            R,
            S,
            clean_ancillae=clean_ancillae,
            ctrls=ctrls,
        )

        adder_controls = (ctrls[0] + [block_encoding_ancillae[0]], ctrls[1] + [1])
        gates += add_classical_value_gate_efficient(
            system.bosonic_system[active_mode],
            -R + S,
            clean_ancillae=clean_ancillae,
            ctrls=adder_controls,
        )

        gates.append(cirq.H.on(block_encoding_ancillae[0]))
        return gates

    _test_helper(
        operator,
        maximum_occupation_number,
        expected_rescaling_factor,
        two_bosonic_operators_block_encoding,
    )


from src.lobe.multiplexed_rotations import get_decomposed_multiplexed_rotation_circuit


def _add_multi_bosonic_rotations(
    rotation_qubit,
    bosonic_mode_register,
    creation_exponent=0,
    annihilation_exponent=0,
    clean_ancillae=[],
    ctrls=([], []),
    numerics=None,
):
    """Add rotations to pickup bosonic coefficients corresponding to a series of ladder operators (assumed
        to be normal ordered) acting on one bosonic mode within a term.

    Args:
        rotation_qubit (cirq.LineQubit): The qubit that is rotated to pickup the amplitude corresponding
            to the coefficients that appear when a bosonic op hits a quantum state
        bosonic_mode_register (List[cirq.LineQubit]): The qubits that store the occupation of the bosonic
            mode being acted upon.
        creation_exponent (int): The number of subsequent creation operators in the term
        annihilation_exponent (int): The number of subsequent annihilation operators in the term
        ctrls (Tuple(List[cirq.LineQubit], List[int])): A set of qubits and integers that correspond to
            the control qubits and values.

    Returns:
        - The gates to perform the unitary operation
    """
    gates = []

    maximum_occupation_number = (1 << len(bosonic_mode_register)) - 1

    # Flip the rotation qubit outside the encoded subspace
    gates.append(
        cirq.Moment(
            cirq.X.on(rotation_qubit).controlled_by(*ctrls[0], control_values=ctrls[1])
        )
    )

    # Multiplexing over computational basis states of mode register that will not be zeroed-out
    angles = []
    for particle_number in range(
        0,
        maximum_occupation_number + 1,
    ):
        if (particle_number - creation_exponent) < 0:
            angles.append(0)
        elif (
            particle_number - creation_exponent + annihilation_exponent
        ) > maximum_occupation_number:
            angles.append(0)
        else:
            # Classically compute coefficient that should appear
            intended_coefficient = 1
            for power in range(creation_exponent):
                intended_coefficient *= np.sqrt(
                    (particle_number - power) / (maximum_occupation_number + 1)
                )
            for power in range(annihilation_exponent):
                intended_coefficient *= np.sqrt(
                    (particle_number - creation_exponent + power + 1)
                    / (maximum_occupation_number + 1)
                )
            angles.append(2 * np.arcsin(-1 * intended_coefficient) / np.pi)

    gates += get_decomposed_multiplexed_rotation_circuit(
        bosonic_mode_register + [rotation_qubit],
        angles,
        clean_ancillae=clean_ancillae,
        ctrls=ctrls,
        numerics=numerics,
    )
    return gates
