import cirq
import numpy as np
from .incrementer import add_incrementer
from .numerical_comparator import is_less_than, is_greater_than
from openparticle import BosonOperator, FermionOperator, AntifermionOperator


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
        bosonic_rotation_index = 0
        if perform_coefficient_oracle:
            bosonic_rotation_index += 1
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

        # # Left-elbow based on system state
        # circuit_ops, system_ctrls, number_of_bosonic_ancillae = _get_system_ctrls(
        #     system,
        #     # clean_ancillae[clean_ancillae_counter],
        #     term,
        #     clean_ancillae[clean_ancillae_counter:],
        # )
        # bosonic_ancillae = clean_ancillae[
        #     clean_ancillae_counter : clean_ancillae_counter + number_of_bosonic_ancillae
        # ]
        # gates_for_term += circuit_ops
        # clean_ancillae_counter += number_of_bosonic_ancillae
        system_ctrls = ([], [])

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

        for operator in term.split()[::-1]:
            if isinstance(operator, BosonOperator):
                if operator.creation:

                    circuit_ops = _update_system(
                        operator,
                        system,
                        clean_ancillae=clean_ancillae[clean_ancillae_counter:],
                        ctrls=([control_qubit], [1]),
                    )
                    gates_for_term += circuit_ops

                    circuit_ops, number_of_bosonic_rotations = _add_bosonic_rotations(
                        system,
                        rotation_register[bosonic_rotation_index:],
                        operator,
                        creation_ops=True,
                        ctrls=([control_qubit], [1]),
                    )
                    gates_for_term += circuit_ops
                    bosonic_rotation_index += number_of_bosonic_rotations
                else:
                    circuit_ops, number_of_bosonic_rotations = _add_bosonic_rotations(
                        system,
                        rotation_register[bosonic_rotation_index:],
                        operator,
                        annihilation_ops=True,
                        ctrls=([control_qubit], [1]),
                    )
                    gates_for_term += circuit_ops
                    bosonic_rotation_index += number_of_bosonic_rotations

                    circuit_ops = _update_system(
                        operator,
                        system,
                        clean_ancillae=clean_ancillae[clean_ancillae_counter:],
                        ctrls=([control_qubit], [1]),
                    )
                    gates_for_term += circuit_ops

            else:
                circuit_ops = _update_system(
                    operator,
                    system,
                    clean_ancillae=clean_ancillae[clean_ancillae_counter:],
                    ctrls=([control_qubit], [1]),
                )
                gates_for_term += circuit_ops

                if isinstance(operator, FermionOperator):
                    system_qubit = system.fermionic_register[operator.mode]
                else:
                    system_qubit = system.antifermionic_register[operator.mode]

                if operator.creation:
                    control_values = [1, 0]
                else:
                    control_values = [1, 1]

                gates_for_term.append(
                    cirq.Moment(
                        cirq.X.on(
                            rotation_register[bosonic_rotation_index]
                        ).controlled_by(
                            control_qubit, system_qubit, control_values=control_values
                        )
                    )
                )

                bosonic_rotation_index += 1

        gates_for_term.append(
            cirq.Moment(cirq.X.on(validation).controlled_by(control_qubit))
        )

        # # Right-elbow to uncompute system qbool assuming term did not fire
        # circuit_ops, system_ctrls, number_of_bosonic_ancillae = _get_system_ctrls(
        #     system,
        #     # clean_ancillae[clean_ancillae_counter],
        #     term,
        #     bosonic_ancillae,
        #     uncompute=True,
        #     ctrls=([control_qubit], [0]),
        # )
        # gates_for_term += circuit_ops
        # clean_ancillae_counter -= number_of_bosonic_ancillae

        gates_for_term.append(
            cirq.Moment(
                cirq.X.on(control_qubit).controlled_by(
                    validation, *index_ctrls[0], control_values=[0] + index_ctrls[1]
                )
            )
        )
        clean_ancillae_counter -= 1

        if perform_coefficient_oracle:
            sign = 1
            if term.coeff < 0:
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
            gates_for_term.append(
                cirq.Moment(
                    cirq.ry(sign * 2 * np.arccos(np.abs(term.coeff)))
                    .on(rotation_register[0])
                    .controlled_by(*index_ctrls[0], control_values=index_ctrls[1])
                )
            )

        # Right-elbow to uncompute index of term
        circuit_ops, _, number_of_used_ancillae = _get_index_register_ctrls(
            index_register,
            clean_ancillae[clean_ancillae_counter:],
            index,
            uncompute=True,
            decompose=decompose,
        )
        clean_ancillae_counter -= number_of_used_ancillae
        gates_for_term += circuit_ops

        all_gates += gates_for_term

        assert clean_ancillae_counter == 0

    return all_gates


def _get_index_register_ctrls(
    index_register, ancillae, index, uncompute=False, decompose=True
):
    """Create a quantum Boolean that stores whether or not the index_register is in the state |index>

    This function operates as an N-Qubit left-elbow gate (Toffoli) controlled on the state of
        index_register and acting onto an ancilla qubit that is promised to begin in the |0> state.

    Args:
        index_register (List[cirq.LineQubit]): The qubit register on which to control
        ancillae (cirq.LineQubit): The ancilla qubit that will store the quantum boolean on output
        index (int): The computational basis state of index_register to control on. Stored as an integer
            and the binary representation gives the control structure on index_register
        uncompute (boolean): A classical flag to dictate whether or not this is a left or right elbow

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
    additional_ancillae=[],
    uncompute=False,
    ctrls=([], []),
):
    """Create a quantum Boolean that stores if the system will be acted on nontrivially by the term.

    This function operates as an N-Qubit left-elbow gate (Toffoli) controlled on the state of
        system and acting onto an ancilla qubit that is promised to begin in the |0> state.

    TODO: For now, we assume that the maximum power on any bosonic particle operator is 1.

    Args:
        system (System): The qubit registers representing the system
        ancilla (cirq.LineQubit): The ancilla qubit that will store the quantum boolean on output
        term (ParticleOperator/ParticleOperatorSum): The term in question
        additional_ancillae (List[cirq.LineQubit]): A list of qubits that are promised to start and in the 0-state.
            They are to be used as ancillae to store quantum booleans regarding the bosonic operators and are
            reset when the uncompute function is called.
        uncompute (boolean): A classical flag to dictate whether or not this is a left or right elbow
        ctrls (Tuple(List[cirq.LineQubit], List[int])): A set of qubits and integers that correspond to
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

    for particle_operator in term.split():

        if type(particle_operator) in [
            FermionOperator,
            AntifermionOperator,
        ]:  # Fermionic or Antifermionic
            if particle_operator.creation:
                control_values.append(0)
            else:
                control_values.append(1)

            if isinstance(particle_operator, FermionOperator):  # Fermionic
                control_qubits.append(system.fermionic_register[particle_operator.mode])
            else:
                control_qubits.append(
                    system.antifermionic_register[particle_operator.mode]
                )

    if uncompute:
        gates = gates[::-1]

    return gates, (control_qubits, control_values), ancillae_counter


def _add_bosonic_rotations(
    system,
    rotation_qubits,
    term,
    creation_ops=False,
    annihilation_ops=False,
    ctrls=([], []),
):
    """Add rotations to pickup bosonic annihilation coefficients.

    Args:
        system (System): The qubit registers representing the system
        rotation_qubits (List[cirq.LineQubit]): A list of qubits that can be rotated to pickup the
            amplitudes corresponding to the coefficients that appear when a bosonic annihilation op
            hits a quantum state
        term (ParticleOperator/ParticleOperatorSum): The term in question
        creation_ops (bool): Dictates whether or not to perform rotations of creation op
        annihilation_ops (bool): Dictates whether or not to perform rotations of annihilation op
        ctrls (Tuple(List[cirq.LineQubit], List[int])): A set of qubits and integers that correspond to
            the control qubits and values.

    Returns:
        - The gates to perform the unitary operation
        - The number of bosonic operators that were accounted for
    """
    gates = []
    number_of_bosonic_ops = 0
    for particle_operator in term.split():
        rotate_for_this_op = False
        if isinstance(particle_operator, BosonOperator):
            if particle_operator.creation and creation_ops:
                rotate_for_this_op = True
            if (not particle_operator.creation) and annihilation_ops:
                rotate_for_this_op = True

        if rotate_for_this_op:

            # Multiplexing over computational basis states of mode register
            for particle_number in range(0, system.maximum_occupation_number + 1):
                occupation_qubits = system.bosonic_system[particle_operator.mode]
                occupation_control_values = [
                    int(i)
                    for i in format(particle_number, f"#0{2+len(occupation_qubits)}b")[
                        2:
                    ]
                ]

                # Rotate ancilla by sqrt(particle_number)
                intended_coefficient = np.sqrt(
                    particle_number / (system.maximum_occupation_number + 1)
                )
                gates.append(
                    cirq.Moment(
                        cirq.ry(2 * np.arccos(intended_coefficient))
                        .on(rotation_qubits[number_of_bosonic_ops])
                        .controlled_by(
                            *occupation_qubits,
                            *ctrls[0],
                            control_values=occupation_control_values + ctrls[1],
                        )
                    )
                )

            number_of_bosonic_ops += 1

    return gates, number_of_bosonic_ops


def _update_system(term, system, clean_ancillae=[], ctrls=([], [])):
    """Add rotations to pickup bosonic annihilation coefficients.

    Args:
        term (ParticleOperator/ParticleOperatorSum): The term in question
        system (System): The qubit registers representing the system
        clean_ancillae (List[cirq.LineQubit]): A list of qubits that are promised to start and end in the 0-state.
            They are to be used as ancillae to store quantum booleans.
        ctrls (Tuple(List[cirq.LineQubit], List[int])): A set of qubits and integers that correspond to
            the control qubits and values.

    Returns:
        - The gates to perform the unitary operation
    """
    gates = []

    for particle_operator in term.split():
        mode = particle_operator.mode
        if isinstance(particle_operator, FermionOperator) or isinstance(
            particle_operator, AntifermionOperator
        ):

            if isinstance(particle_operator, FermionOperator):
                register = system.fermionic_register
            else:
                register = system.antifermionic_register

            # parity constraint
            for system_qubit in register[::-1][:mode]:
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
                    cirq.X.on(register[-mode - 1]).controlled_by(
                        *ctrls[0], control_values=ctrls[1]
                    )
                )
            )
        else:
            gates += add_incrementer(
                [],
                system.bosonic_system[mode],
                clean_ancillae[: len(system.bosonic_system[mode]) - 2],
                not (particle_operator.creation),
                ctrls[0],
                ctrls[1],
            )

    return gates
