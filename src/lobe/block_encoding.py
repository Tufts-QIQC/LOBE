import cirq
import numpy as np
from .incrementer import add_classical_value
from openparticle import (
    BosonOperator,
    FermionOperator,
    AntifermionOperator,
)
from ._utils import get_parsed_dictionary


def add_lobe_oracle(
    operators,
    validation,
    index_register,
    system,
    rotation_register,
    clean_ancillae=[],
    perform_coefficient_oracle=True,
    decompose=True,
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
        decompose (bool): Classical boolean determining if quantum conditions should be decomposed into ancillae
            to reduce number of Toffolis (True) or if they should be left as a series of multiple controls to reduce
            additional ancillae requirements (False).

    Returns:
        - The gates to perform the LOBE oracle
    """
    all_gates = []
    clean_ancillae_counter = 0

    for index, term in enumerate(operators):
        # assert term.is_normal_ordered(term.split())

        bosonic_rotation_register = rotation_register
        if perform_coefficient_oracle:
            bosonic_rotation_register = rotation_register[1:]

        gates_for_term = []

        control_qubit = clean_ancillae[clean_ancillae_counter]
        clean_ancillae_counter += 1

        # Left-elbow based on index of term
        circuit_ops, index_ctrls, number_of_used_ancillae = _get_index_register_ctrls(
            index_register,
            clean_ancillae[clean_ancillae_counter:],
            index,
            decompose=decompose,
        )
        gates_for_term += circuit_ops
        clean_ancillae_counter += number_of_used_ancillae

        # Left-elbow based on system state
        circuit_ops, system_ctrls, number_of_bosonic_ancillae = _get_system_ctrls(
            system,
            term,
        )
        gates_for_term += circuit_ops
        clean_ancillae_counter += number_of_bosonic_ancillae

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
            gates_for_term.append(
                cirq.Moment(
                    cirq.ry(2 * np.arccos(np.abs(term.coeff)))
                    .on(rotation_register[0])
                    .controlled_by(*index_ctrls[0], control_values=index_ctrls[1])
                )
            )

        # Right-elbow to uncompute index of term
        circuit_ops, _, number_of_used_ancillae = _get_index_register_ctrls(
            index_register,
            clean_ancillae[clean_ancillae_counter:],
            index,
            decompose=decompose,
        )
        clean_ancillae_counter -= number_of_used_ancillae
        gates_for_term += circuit_ops

        all_gates += gates_for_term

        assert clean_ancillae_counter == 0

    return all_gates


def _apply_term(
    term, system, clean_ancillae, bosonic_rotation_register, ctrls=([], [])
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

    gates += _update_fermionic_and_antifermionic_system(
        term,
        system,
        ctrls=ctrls,
    )

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
                ctrls=ctrls,
            )
            bosonic_counter += 1
            gates += add_classical_value(
                system.bosonic_system[mode],
                creation_exponent - annihilation_exponent,
                clean_ancillae,
                ctrls=ctrls,
            )

    return gates


def _get_index_register_ctrls(index_register, ancillae, index, decompose=True):
    """Create a quantum Boolean that stores whether or not the index_register is in the state |index>

    This function operates as an N-Qubit left-elbow gate (Toffoli) controlled on the state of
        index_register and acting onto an ancilla qubit that is promised to begin in the |0> state.

    Args:
        index_register (List[cirq.LineQubit]): The qubit register on which to control
        ancillae (cirq.LineQubit): The ancilla qubit that will store the quantum boolean on output
        index (int): The computational basis state of index_register to control on. Stored as an integer
            and the binary representation gives the control structure on index_register
        decompose (bool): Classical boolean determining if quantum conditions should be decomposed into ancillae
            to reduce number of Toffolis (True) or if they should be left as a series of multiple controls to reduce
            additional ancillae requirements (False).

    Returns:
        - The gates to perform the unitary operation
        - The qubit representing the quantum boolean
    """
    gates = []

    # Get binary control values corresponding to index
    index_register_control_values = [
        int(i) for i in format(index, f"#0{2+len(index_register)}b")[2:]
    ]

    if decompose:
        # Elbow onto ancilla: will be |1> iff index_register is in comp. basis state |index>
        gates.append(
            cirq.Moment(
                cirq.X.on(ancillae[0]).controlled_by(
                    *index_register,
                    control_values=index_register_control_values,
                )
            )
        )
        return gates, ([ancillae[0]], [1]), 1

    return gates, (index_register, index_register_control_values), 0


def _get_system_ctrls(
    system,
    term,
    uncompute=False,
):
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
        - The gates to perform the unitary operation
        - The qubit representing the quantum boolean
        - The number of clean ancillae used
    """
    gates = []
    ancillae_counter = 0

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

    if uncompute:
        gates = gates[::-1]

    return gates, (control_qubits, control_values), ancillae_counter


def _add_bosonic_rotations(
    rotation_qubit,
    bosonic_mode_register,
    creation_exponent=0,
    annihilation_exponent=0,
    ctrls=([], []),
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
    for particle_number in range(
        annihilation_exponent,
        min(
            maximum_occupation_number + 1,
            maximum_occupation_number + annihilation_exponent - creation_exponent + 1,
        ),
    ):
        # Get naive controls
        occupation_control_values = [
            int(i)
            for i in format(particle_number, f"#0{2+len(bosonic_mode_register)}b")[2:]
        ]

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

        # Rotate ancilla by theta to pickup desired coefficient in the encoded subspace (|0>)
        gates.append(
            cirq.Moment(
                cirq.ry(2 * np.arcsin(-1 * intended_coefficient))
                .on(rotation_qubit)
                .controlled_by(
                    *bosonic_mode_register,
                    *ctrls[0],
                    control_values=occupation_control_values + ctrls[1],
                )
            )
        )

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

            # update occupation
            gates.append(
                cirq.Moment(
                    cirq.X.on(register[mode]).controlled_by(
                        *ctrls[0], control_values=ctrls[1]
                    )
                )
            )

    return gates
