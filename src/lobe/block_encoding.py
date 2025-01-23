import cirq
import numpy as np
from .addition import add_classical_value_incrementers, _get_p_val
from openparticle import (
    BosonOperator,
    FermionOperator,
    AntifermionOperator,
)
from ._utils import get_parsed_dictionary
from .multiplexed_rotations import get_decomposed_multiplexed_rotation_circuit
import math


def add_lobe_oracle(
    operators,
    validation,
    index_register,
    system,
    rotation_register,
    clean_ancillae=[],
    perform_coefficient_oracle=True,
    numerics=None,
):
    """This function should add the Ladder Operator Block Encoding oracle.

    Args:
        operators (List[ParticleOperator/ParticleOperatorSum]): The ladder operators included in the Hamiltonian.
            Each item in the list is a ParticleOperator and corresponds to a term comprising several
            ladder operators.
        validation (cirq.LineQubit): The validation qubit
        index_register (List[cirq.LineQubit]): The qubit register that is used to index the operators
        system (System): An instance of the System class that holds the qubit registers storing the
            state of the system.
        rotation_register (List[cirq.LineQubit]): The qubit register used to store the coefficients of the terms.
            The qubit at index 0 is used to store the coefficient of the term. The additional qubits are used to
            store the values of the coefficients that get picked up by bosonic ladder operators.
        clean_ancillae (List[cirq.LineQubit]): A list of qubits that are promised to start and end in the 0-state.
            They are to be used as ancillae to store quantum booleans.
        perform_coefficient_oracle (bool): Classical boolean to dictate if the coefficient oracle should be added.

    Returns:
        - The gates to perform the LOBE oracle
    """
    if numerics is None:
        numerics = {}
        numerics["left_elbows"] = 0
        numerics["right_elbows"] = 0
        numerics["ancillae_tracker"] = [
            1 + len(index_register) + len(rotation_register)
        ]
        numerics["rotations"] = 0
        numerics["angles"] = []

    all_gates = []
    clean_ancillae_counter = 0

    # cost of ctrld-multiplexing over the index register
    numerics["left_elbows"] += len(operators)
    numerics["right_elbows"] += len(operators)
    numerics["ancillae_tracker"].append(
        numerics["ancillae_tracker"][-1] + len(index_register)
    )

    for index, term in enumerate(operators):
        # assert term.is_normal_ordered(term.split())

        bosonic_rotation_register = rotation_register
        if perform_coefficient_oracle:
            bosonic_rotation_register = rotation_register[1:]

        gates_for_term = []

        control_qubit = clean_ancillae[clean_ancillae_counter]
        clean_ancillae_counter += 1

        # Left-elbow based on index of term, gate costs have already been accounted for
        circuit_ops, index_ctrls = _get_index_register_ctrls(
            index_register,
            clean_ancillae[clean_ancillae_counter:],
            index,
        )
        gates_for_term += circuit_ops
        clean_ancillae_counter += 1

        system_ctrls = _get_system_ctrls(system, term)

        # Left-elbow onto qubit to mark if term should fire
        gates_for_term.append(
            cirq.Moment(
                cirq.X.on(control_qubit).controlled_by(
                    *system_ctrls[0],
                    *index_ctrls[0],
                    validation,
                    control_values=system_ctrls[1] + index_ctrls[1] + [1],
                )
            )
        )
        # Decomposing N controls into one left-elbow requires N - 1 left-elbows, N - 2 right-elbows,
        # and N - 2 temporary ancilla (one additional ancilla stores the output quantum boolean)
        numerics["left_elbows"] += len(system_ctrls[1] + [1] + [1]) - 1
        numerics["right_elbows"] += len(system_ctrls[1] + [1] + [1]) - 2
        numerics["ancillae_tracker"].append(
            numerics["ancillae_tracker"][-1] + len(system_ctrls[1] + [1] + [1]) - 1
        )
        numerics["ancillae_tracker"].append(numerics["ancillae_tracker"][-2] + 1)

        # Flip validation qubit if term fires
        gates_for_term.append(
            cirq.Moment(cirq.X.on(validation).controlled_by(control_qubit))
        )

        # Apply term onto system
        gates_for_term += _apply_term(
            term,
            system,
            clean_ancillae[clean_ancillae_counter:],
            bosonic_rotation_register,
            ctrls=([control_qubit], [1]),
            numerics=numerics,
        )

        # Uncompute control qubit
        gates_for_term.append(
            cirq.Moment(
                cirq.X.on(control_qubit).controlled_by(
                    validation, *index_ctrls[0], control_values=[0] + index_ctrls[1]
                )
            )
        )
        clean_ancillae_counter -= 1
        numerics["left_elbows"] += 1
        numerics["right_elbows"] += 1
        numerics["ancillae_tracker"].append(numerics["ancillae_tracker"][-1] + 1)
        numerics["ancillae_tracker"].append(numerics["ancillae_tracker"][-1] - 1)
        numerics["ancillae_tracker"].append(numerics["ancillae_tracker"][-1] - 1)

        if term.coeff < 0:
            if len(index_ctrls[0]) == 1:
                if index_ctrls[1][0] == 0:
                    gates_for_term.append(cirq.Moment(cirq.X.on(index_ctrls[0][0])))
                gates_for_term.append(cirq.Moment(cirq.Z.on(index_ctrls[0][0])))
                if index_ctrls[1][0] == 0:
                    gates_for_term.append(cirq.Moment(cirq.X.on(index_ctrls[0][0])))
            else:
                # get a negative 1 coeff by using pauli algebra to get a -Identity on the rotation qubit
                gates_for_term.append(
                    cirq.Moment(
                        cirq.X.on(rotation_register[0]).controlled_by(
                            *index_ctrls[0], control_values=index_ctrls[1]
                        )
                    )
                )
                gates_for_term.append(
                    cirq.Moment(
                        cirq.Z.on(rotation_register[0]).controlled_by(
                            *index_ctrls[0], control_values=index_ctrls[1]
                        )
                    )
                )
                gates_for_term.append(
                    cirq.Moment(
                        cirq.X.on(rotation_register[0]).controlled_by(
                            *index_ctrls[0], control_values=index_ctrls[1]
                        )
                    )
                )
                gates_for_term.append(
                    cirq.Moment(
                        cirq.Z.on(rotation_register[0]).controlled_by(
                            *index_ctrls[0], control_values=index_ctrls[1]
                        )
                    )
                )

        # Perform rotations related to term coefficients
        if perform_coefficient_oracle:
            rotation_angle = 2 * np.arccos(np.abs(term.coeff))
            if math.isnan(rotation_angle):
                print(
                    "Encountered nan for rotation angle. Term coeff was: {}".format(
                        term.coeff
                    )
                )
                rotation_angle = np.sqrt(2)
            gates_for_term.append(
                cirq.Moment(
                    cirq.ry(rotation_angle)
                    .on(rotation_register[0])
                    .controlled_by(*index_ctrls[0], control_values=index_ctrls[1])
                )
            )
            numerics["rotations"] += 2
            numerics["angles"].append(-rotation_angle / 2)
            numerics["angles"].append(rotation_angle / 2)

        # Right-elbow to uncompute index of term
        circuit_ops, _ = _get_index_register_ctrls(
            index_register,
            clean_ancillae[clean_ancillae_counter:],
            index,
        )
        gates_for_term += circuit_ops
        clean_ancillae_counter -= 1

        all_gates += gates_for_term

        assert clean_ancillae_counter == 0

    return all_gates


def _apply_term(
    term,
    system,
    clean_ancillae,
    bosonic_rotation_register,
    ctrls=([], []),
    numerics=None,
):
    """Apply a single term to the state of the system and apply bosonic coefficient rotations.

    Args:
        term (ParticleOperator/ParticleOperatorSum): The term in question
        system (System): The qubit registers representing the system
        clean_ancillae (List[cirq.LineQubit]): A list of qubits that are promised to start and end in the 0-state.
            They are to be used as ancillae to store quantum booleans.
        bosonic_rotation_register (List[cirq.LineQubit]): A list of qubits that are rotated corresponding
            to the coefficients that are picked up by the bosonic operators.
        ctrls (Tuple(List[cirq.LineQubit], List[int])): A set of qubits and integers that correspond to
            the control qubits and values.

    Returns:
        - The gates to perform the unitary operation
    """
    gates = []
    # assert term.is_normal_ordered(term.split())
    operator_dictionary = get_parsed_dictionary(term, system.number_of_modes)

    gates += _update_fermionic_and_antifermionic_system(term, system, ctrls=ctrls)

    bosonic_counter = 0
    # Bosonic Ladder Operators
    for mode in range(system.number_of_modes):
        creation_exponent = operator_dictionary["boson"]["creation"][mode]
        annihilation_exponent = operator_dictionary["boson"]["annihilation"][mode]

        if not ((creation_exponent == 0) and (annihilation_exponent == 0)):
            gates += _add_bosonic_rotations(
                bosonic_rotation_register[bosonic_counter],
                system.bosonic_system[mode],
                creation_exponent,
                annihilation_exponent,
                clean_ancillae=clean_ancillae,
                ctrls=ctrls,
                numerics=numerics,
            )
            bosonic_counter += 1
            adder_gates, _ = add_classical_value_incrementers(
                system.bosonic_system[mode],
                creation_exponent - annihilation_exponent,
                clean_ancillae,
                ctrls=ctrls,
            )
            gates += adder_gates
            p_val = _get_p_val(
                creation_exponent - annihilation_exponent,
                len(system.bosonic_system[mode]),
            )

            numerics["left_elbows"] += (
                len(system.bosonic_system[mode]) - p_val - 1
            )  # N - p - 1 elbows
            numerics["right_elbows"] += len(system.bosonic_system[mode]) - p_val - 1
            numerics["ancillae_tracker"].append(  # 2 (N - p) - 1 temporary ancillae
                numerics["ancillae_tracker"][-1]
                + 2 * len(system.bosonic_system[mode])
                - 2 * p_val
                - 1
            )
            numerics["ancillae_tracker"].append(numerics["ancillae_tracker"][-2])

    return gates


def _get_index_register_ctrls(index_register, ancillae, index):
    """Create a quantum Boolean that stores whether or not the index_register is in the state |index>

    This function operates as an N-Qubit left-elbow gate (Toffoli) controlled on the state of
        index_register and acting onto an ancilla qubit that is promised to begin in the |0> state.

    Args:
        index_register (List[cirq.LineQubit]): The qubit register on which to control
        ancillae (cirq.LineQubit): The ancilla qubit that will store the quantum boolean on output
        index (int): The computational basis state of index_register to control on. Stored as an integer
            and the binary representation gives the control structure on index_register

    Returns:
        - The gates to perform the unitary operation
        - The qubit representing the quantum boolean
    """
    gates = []

    # Get binary control values corresponding to index
    index_register_control_values = [
        int(i) for i in format(index, f"#0{2+len(index_register)}b")[2:]
    ]

    # Elbow onto ancilla: will be |1> iff index_register is in comp. basis state |index>
    gates.append(
        cirq.Moment(
            cirq.X.on(ancillae[0]).controlled_by(
                *index_register,
                control_values=index_register_control_values,
            )
        )
    )
    return gates, ([ancillae[0]], [1])


def _get_system_ctrls(system, term, uncompute=False):
    """Create a quantum Boolean that stores if the system will be acted on nontrivially by the term.

    This function operates as an N-Qubit left-elbow gate (Toffoli) controlled on the state of
        system and acting onto an ancilla qubit that is promised to begin in the |0> state.

    TODO: For now, we assume that the maximum power on any bosonic particle operator is 1.

    Args:
        system (System): The qubit registers representing the system
        ancilla (cirq.LineQubit): The ancilla qubit that will store the quantum boolean on output
        term (ParticleOperator/ParticleOperatorSum): The term in question
        uncompute (boolean): A classical flag to dictate whether or not this is a left or right elbow
            the control qubits and values.

    Returns:
        - The qubit representing the quantum boolean
        - The number of clean ancillae used
    """
    control_qubits = []
    control_values = []

    for particle_operator in term.split()[::-1]:

        if type(particle_operator) in [
            FermionOperator,
            AntifermionOperator,
        ]:  # Fermionic or Antifermionic
            if particle_operator.creation:
                control_values.append(0)
            else:
                control_values.append(1)

            if isinstance(particle_operator, FermionOperator):  # Fermionic
                qubit = system.fermionic_register[particle_operator.mode]
            else:
                qubit = system.antifermionic_register[particle_operator.mode]

            if qubit in control_qubits:
                # mode is already being acted upon. We assume this means that we're
                #   looking at a number operator in the term so we just want to control
                #   on the mode being occupied
                control_values = control_values[:-1]
                control_values[control_qubits.index(qubit)] = 1
            else:
                control_qubits.append(qubit)
        else:
            if type(particle_operator) != BosonOperator:
                raise RuntimeError(
                    "unknown particle type: {}".format(particle_operator.particle_type)
                )

    return (control_qubits, control_values)


def _add_bosonic_rotations(
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
        min(
            maximum_occupation_number + 1,
            maximum_occupation_number + annihilation_exponent - creation_exponent + 1,
        ),
    ):
        if particle_number < annihilation_exponent:
            angles.append(0)
        else:
            # Classically compute coefficient that should appear
            intended_coefficient = 1
            for power in range(annihilation_exponent):
                intended_coefficient *= np.sqrt(
                    (particle_number - power) / (maximum_occupation_number + 1)
                )
            for power in range(creation_exponent):
                intended_coefficient *= np.sqrt(
                    (particle_number - annihilation_exponent + power + 1)
                    / (maximum_occupation_number + 1)
                )
            angles.append(2 * np.arcsin(-1 * intended_coefficient) / np.pi)

    rotation_gates, rotation_metrics = get_decomposed_multiplexed_rotation_circuit(
        bosonic_mode_register,
        rotation_qubit,
        angles,
        clean_ancillae=clean_ancillae,
        ctrls=ctrls,
    )
    gates += rotation_gates
    if numerics is not None:
        numerics["rotations"] += rotation_metrics.number_of_nonclifford_rotations
    return gates


def _update_fermionic_and_antifermionic_system(term, system, ctrls=([], [])):
    """Update the fermionic and antifermionic system corresponding to the operators in the term.

    Args:
        term (ParticleOperator/ParticleOperatorSum): The term in question
        system (System): The qubit registers representing the system
        ctrls (Tuple(List[cirq.LineQubit], List[int])): A set of qubits and integers that correspond to
            the control qubits and values.

    Returns:
        - The gates to perform the unitary operation
    """
    gates = []

    for particle_operator in term.split()[::-1]:
        mode = particle_operator.mode
        if isinstance(particle_operator, FermionOperator) or isinstance(
            particle_operator, AntifermionOperator
        ):

            if isinstance(particle_operator, FermionOperator):
                register = system.fermionic_register
            else:
                register = system.antifermionic_register

            # parity constraint
            for system_qubit in register[:mode]:
                gates.append(
                    cirq.Moment(
                        cirq.Z.on(system_qubit).controlled_by(
                            *ctrls[0], control_values=ctrls[1]
                        )
                    )
                )

            if isinstance(particle_operator, AntifermionOperator):
                # Additional (-1)**(N_fermions in state) parity constraint when acting on state with antifermionic operator
                for (
                    system_qubit
                ) in (
                    system.fermionic_register
                ):  # Place Z's on fermionic occupancy register
                    gates.append(
                        cirq.Moment(
                            cirq.Z.on(system_qubit).controlled_by(
                                *ctrls[0], control_values=ctrls[1]
                            )
                        )
                    )

            # update occupation
            gates.append(
                cirq.Moment(
                    cirq.X.on(register[mode]).controlled_by(
                        *ctrls[0], control_values=ctrls[1]
                    )
                )
            )

    return gates
