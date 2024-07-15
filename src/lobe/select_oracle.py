import cirq
import numpy as np
from .incrementer import add_incrementer


def add_select_oracle(
    circuit,
    validation,
    index_register,
    system,
    operators,
    bosonic_rotation_register=[],
    clean_ancilla=[],
):
    """Add the select oracle for LOBE into the quantum circuit.

    Args:
        circuit (cirq.Circuit): The quantum circuit onto which the select oracle will be added
        validation (cirq.LineQubit): The validation qubit
        index_register (List[cirq.LineQubit]): The qubit register that is used to index the operators
        system (System): An instance of the System class that holds the qubit registers storing the
            state of the system.
        operators (List[List[LadderOperator]]): The ladder operators included in the Hamiltonian.
            Each item in the list is a list of LadderOperators and corresponds to a term comprising several
            ladder operators.

    Returns:
        cirq.Circuit: The updated quantum circuit
    """
    for operator_index, operator in enumerate(operators):
        op_types = [ladder_op.particle_type for ladder_op in operator]
        if np.allclose(op_types, 0):  # All terms are fermionic terms
            if len(operator) == 2 and operator[0].mode == operator[1].mode:
                circuit = _add_fermionic_particle_number_op(
                    circuit,
                    validation,
                    index_register,
                    operator_index,
                    system.fermionic_register[-operator[0].mode - 1],
                )
            else:
                circuit = _add_fermionic_ladder_operator(
                    circuit,
                    validation,
                    index_register,
                    operator_index,
                    system.fermionic_register,
                    operator,
                    clean_ancilla,
                )
        elif np.allclose(op_types, 2):  # All terms are bosonic terms
            if len(operator) == 2 and operator[0].mode == operator[1].mode:
                circuit = _add_bosonic_particle_number_op(
                    circuit,
                    validation,
                    index_register,
                    operator_index,
                    operator,
                    system,
                    bosonic_rotation_register,
                    clean_ancilla,
                )
            else:
                circuit = _add_bosonic_ladder_operator(
                    circuit,
                    validation,
                    index_register,
                    operator_index,
                    system,
                    operator,
                    bosonic_rotation_register,
                    clean_ancilla,
                )
    return circuit


def _add_bosonic_particle_number_op(
    circuit,
    validation,
    index_register,
    operator_index,
    operator,
    system,
    bosonic_rotation_register,
    clean_ancilla,
):
    # Get binary control values corresponding to current operator index
    index_register_control_values = [
        int(i) for i in format(operator_index, f"#0{2+len(index_register)}b")[2:]
    ]

    # left elbow onto clean_ancilla[0]: will be |1> iff occupation is 0
    circuit.append(
        cirq.Moment(
            cirq.X.on(clean_ancilla[0]).controlled_by(
                *system.bosonic_system[operator[0].mode],
                control_values=[0] * len(system.bosonic_system[operator[0].mode]),
            )
        )
    )

    control_values = index_register_control_values + [0]
    control_qubits = index_register + [clean_ancilla[0]]

    # Perform two subsequent rotations to pickup a total coefficient of sqrt(N_p)/Omega
    circuit = _add_bosonic_coefficient_rotation(
        circuit,
        system.bosonic_system[operator[0].mode],
        bosonic_rotation_register[0],
        ctrls=control_qubits + [validation],
        ctrl_values=control_values + [1],
    )
    circuit = _add_bosonic_coefficient_rotation(
        circuit,
        system.bosonic_system[operator[0].mode],
        bosonic_rotation_register[1],
        ctrls=control_qubits + [validation],
        ctrl_values=control_values + [1],
    )

    # Flip validation to 0 to mark that we hit a state that statisfied this operator
    circuit.append(
        cirq.Moment(
            cirq.X.on(validation).controlled_by(
                *control_qubits, control_values=control_values
            )
        )
    )

    # right elbow to reset clean_ancilla[0]
    circuit.append(
        cirq.Moment(
            cirq.X.on(clean_ancilla[0]).controlled_by(
                *system.bosonic_system[operator[0].mode],
                control_values=[0] * len(system.bosonic_system[operator[0].mode]),
            )
        )
    )
    return circuit


def _add_bosonic_ladder_operator(
    circuit,
    validation,
    index_register,
    operator_index,
    system,
    operator,
    bosonic_rotation_register,
    clean_ancilla,
):
    # Get binary control values corresponding to current operator index
    index_register_control_values = [
        int(i) for i in format(operator_index, f"#0{2+len(index_register)}b")[2:]
    ]
    control_values = [1] + index_register_control_values  # validation
    control_qubits = [validation] + index_register

    # left-elbows onto various ancilla
    used_ancilla = 0
    for i, ladder_op in enumerate(operator[::-1]):

        if (
            ladder_op.creation
        ):  # creation op means particle number should be less than omega
            current_control_val = 1
        else:  # annihilation op means particle number should be greater than 0
            current_control_val = 0

        # left-elbow onto clean_ancilla[i]: will be |0> if occupation is suitable
        circuit.append(
            cirq.Moment(
                cirq.X.on(clean_ancilla[i]).controlled_by(
                    *system.bosonic_system[ladder_op.mode]
                    + [validation]
                    + index_register,
                    control_values=[current_control_val]
                    * len(system.bosonic_system[ladder_op.mode])
                    + [1]
                    + index_register_control_values,
                )
            )
        )

        control_qubits += [clean_ancilla[i]]
        control_values += [0]
        used_ancilla += 1

    # Reverse loop because operators act starting from the right
    bosonic_rotation_counter = 0
    for ladder_op in operator[::-1]:
        if not ladder_op.creation:
            _add_bosonic_coefficient_rotation(
                circuit,
                system.bosonic_system[ladder_op.mode],
                bosonic_rotation_register[bosonic_rotation_counter],
                ctrls=control_qubits,
                ctrl_values=control_values,
            )
            bosonic_rotation_counter += 1
        # perform incrementers/decrementers
        circuit = add_incrementer(
            circuit,
            system.bosonic_system[ladder_op.mode],
            clean_ancilla[used_ancilla:],
            not ladder_op.creation,
            control_qubits,
            control_values,
        )
        if ladder_op.creation:
            _add_bosonic_coefficient_rotation(
                circuit,
                system.bosonic_system[ladder_op.mode],
                bosonic_rotation_register[bosonic_rotation_counter],
                ctrls=control_qubits,
                ctrl_values=control_values,
            )
            bosonic_rotation_counter += 1

    # Mark validation qubit if term fired
    circuit.append(
        cirq.Moment(
            cirq.X.on(validation).controlled_by(
                *control_qubits[1:], control_values=control_values[1:]
            )
        )
    )

    # right-elbows to clean various ancilla
    for i, ladder_op in enumerate(operator):

        if (
            ladder_op.creation
        ):  # creation op means particle number should be less than omega
            current_control_val = 1
        else:  # annihilation op means particle number should be greater than 0
            current_control_val = 0

        # right-elbow to clean clean_ancilla[i]
        circuit.append(
            cirq.Moment(
                cirq.X.on(clean_ancilla[used_ancilla - 1]).controlled_by(
                    *(
                        system.bosonic_system[ladder_op.mode]
                        + [validation]
                        + index_register
                    ),
                    control_values=(
                        [current_control_val]
                        * len(system.bosonic_system[ladder_op.mode])
                    )
                    + [1]
                    + index_register_control_values,
                )
            )
        )

        used_ancilla -= 1

    assert used_ancilla == 0

    return circuit


def _add_bosonic_coefficient_rotation(
    circuit, bosonic_mode, rotation_qubit, ctrls=[], ctrl_values=[]
):
    # Multiplexing over computational basis states of mode register
    for particle_number in range(1, 1 << len(bosonic_mode)):
        bosonic_register_control_values = [
            int(i) for i in format(particle_number, f"#0{2+len(bosonic_mode)}b")[2:]
        ]

        # Rotate ancilla by sqrt(particle_number)
        omega = 1 << len(bosonic_mode)
        intended_coefficient = np.sqrt(particle_number / omega)
        circuit.append(
            cirq.Moment(
                cirq.ry(2 * np.arccos(intended_coefficient))
                .on(rotation_qubit)
                .controlled_by(
                    *(bosonic_mode + ctrls),
                    control_values=bosonic_register_control_values + ctrl_values,
                )
            )
        )

    return circuit


def _add_fermionic_particle_number_op(
    circuit, validation, index_register, operator_index, system_qubit
):
    # Get binary control values corresponding to current operator index
    index_register_control_values = [
        int(i) for i in format(operator_index, f"#0{2+len(index_register)}b")[2:]
    ]
    control_values = index_register_control_values + [1]
    control_qubits = index_register + [system_qubit]
    # Flip validation to 0 to mark that we hit a state that statisfied this operator
    circuit.append(
        cirq.X.on(validation).controlled_by(
            *control_qubits, control_values=control_values
        )
    )
    return circuit


def _add_fermionic_ladder_operator(
    circuit, validation, index_register, operator_index, system, operator, clean_ancilla
):
    # Get binary control values corresponding to current operator index
    index_register_control_values = [
        int(i) for i in format(operator_index, f"#0{2+len(index_register)}b")[2:]
    ]
    control_values = (
        [1]
        + index_register_control_values
        + (
            [0] * (len(operator) // 2)
        )  # Make sure system qubits for creation ops are 0-ctrl's
        + (
            [1] * (len(operator) // 2)
        )  # Make sure system qubits for annihilation ops are 1-ctrl's
    )
    control_qubits = [validation] + index_register
    # Add system qubits to control register
    for ladder_op in operator:
        control_qubits += [system[-ladder_op.mode - 1]]

    # This is essentially a left-elbow
    circuit.append(
        cirq.X.on(clean_ancilla[0]).controlled_by(
            *control_qubits, control_values=control_values
        )
    )

    # Reverse loop because operators act starting from the right
    for ladder_op in operator[::-1]:
        for system_qubit in system[::-1][: ladder_op.mode]:
            circuit.append(cirq.Z.on(system_qubit).controlled_by(clean_ancilla[0]))
        circuit.append(
            cirq.X.on(system[-ladder_op.mode - 1]).controlled_by(clean_ancilla[0])
        )

    circuit.append(cirq.X.on(validation).controlled_by(clean_ancilla[0]))

    # This is essentially a right-elbow
    circuit.append(
        cirq.X.on(clean_ancilla[0]).controlled_by(
            *([validation] + index_register),
            control_values=[0] + index_register_control_values,
        )
    )

    return circuit
