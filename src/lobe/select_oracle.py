import cirq
import numpy as np
from .incrementer import add_incrementer
from openparticle import ParticleOperator, ParticleOperatorSum


def add_select_oracle(
    circuit, validation, index_register, system, operators, clean_ancilla=[]
):
    """Add the select oracle for LOBE into the quantum circuit.

    Args:
        circuit (cirq.Circuit): The quantum circuit onto which the select oracle will be added
        validation (cirq.LineQubit): The validation qubit
        index_register (List[cirq.LineQubit]): The qubit register that is used to index the operators
        system (System): An instance of the System class that holds the qubit registers storing the
            state of the system.
        operators (List[ParticleOperator/ParticleOperatorSum]): The ladder operators included in the Hamiltonian.
            Each item in the list is a ParticleOperator and corresponds to a term comprising several
            ladder operators.

    Returns:
        cirq.Circuit: The updated quantum circuit
    """

    if isinstance(operators, ParticleOperatorSum):
        operators = operators.to_list()
    elif isinstance(operators, ParticleOperator):
        operators = [operators]

    for operator_index, operator in enumerate(operators):
        op_types = operator.particle_type
        if all(char == "b" for char in op_types):  # All terms are fermionic terms
            modes = operator.modes
            if len(modes) == 2 and modes[0] == modes[1]:  # b_p^dagger b_q
                circuit = _add_fermionic_particle_number_op(
                    circuit,
                    validation,
                    index_register,
                    operator_index,
                    system.fermionic_register[-modes[0] - 1],
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
        elif all(char == "a" for char in op_types):  # All terms are bosonic terms
            modes = operator.modes
            if len(modes) == 2 and modes[0] == modes[1]:
                circuit = _add_bosonic_particle_number_op(
                    circuit,
                    validation,
                    index_register,
                    operator_index,
                    operator,
                    system,
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
                    clean_ancilla,
                )
    return circuit


def _add_bosonic_particle_number_op(
    circuit, validation, index_register, operator_index, operator, system, clean_ancilla
):
    # Get binary control values corresponding to current operator index
    index_register_control_values = [
        int(i) for i in format(operator_index, f"#0{2+len(index_register)}b")[2:]
    ]

    # left elbow onto clean_ancilla[0]: will be |1> iff occupation is 0
    circuit.append(
        cirq.Moment(
            cirq.X.on(clean_ancilla[0]).controlled_by(
                *system.bosonic_system[operator.modes[0]],
                control_values=[0] * len(system.bosonic_system[operator.modes[0]]),
            )
        )
    )

    control_values = index_register_control_values + [0]
    control_qubits = index_register + [clean_ancilla[0]]
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
                *system.bosonic_system[operator.modes[0]],
                control_values=[0] * len(system.bosonic_system[operator.modes[0]]),
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
    for i, ladder_op in enumerate(operator.split()[::-1]):

        if (
            ladder_op.ca_string == "c"
        ):  # creation op means particle number should be less than omega
            current_control_val = 1
        else:  # annihilation op means particle number should be greater than 0
            current_control_val = 0

        # left-elbow onto clean_ancilla[i]: will be |0> if occupation is suitable
        circuit.append(
            cirq.Moment(
                cirq.X.on(clean_ancilla[i]).controlled_by(
                    *system.bosonic_system[ladder_op.modes[0]]
                    + [validation]
                    + index_register,
                    control_values=[current_control_val]
                    * len(system.bosonic_system[ladder_op.modes[0]])
                    + [1]
                    + index_register_control_values,
                )
            )
        )

        control_qubits += [clean_ancilla[i]]
        control_values += [0]
        used_ancilla += 1

    # Reverse loop because operators act starting from the right
    for ladder_op in operator.split()[::-1]:
        # perform incrementers/decrementers
        circuit = add_incrementer(
            circuit,
            system.bosonic_system[ladder_op.modes[0]],
            clean_ancilla[used_ancilla:],
            not (ladder_op.ca_string == "c"),
            control_qubits,
            control_values,
        )

    # Mark validation qubit if term fired
    circuit.append(
        cirq.Moment(
            cirq.X.on(validation).controlled_by(
                *control_qubits[1:], control_values=control_values[1:]
            )
        )
    )

    # right-elbows to clean various ancilla
    for i, ladder_op in enumerate(operator.split()):

        if (
            ladder_op.ca_string == "c"
        ):  # creation op means particle number should be less than omega
            current_control_val = 1
        else:  # annihilation op means particle number should be greater than 0
            current_control_val = 0

        # right-elbow to clean clean_ancilla[i]
        circuit.append(
            cirq.Moment(
                cirq.X.on(clean_ancilla[used_ancilla - 1]).controlled_by(
                    *(
                        system.bosonic_system[ladder_op.modes[0]]
                        + [validation]
                        + index_register
                    ),
                    control_values=(
                        [current_control_val]
                        * len(system.bosonic_system[ladder_op.modes[0]])
                    )
                    + [1]
                    + index_register_control_values,
                )
            )
        )

        used_ancilla -= 1

    assert used_ancilla == 0

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
            [0] * (len(operator.modes) // 2)
        )  # Make sure system qubits for creation ops are 0-ctrl's
        + (
            [1] * (len(operator.modes) // 2)
        )  # Make sure system qubits for annihilation ops are 1-ctrl's
    )
    control_qubits = [validation] + index_register
    # Add system qubits to control register
    for mode in operator.modes:
        control_qubits += [system[-mode - 1]]

    # This is essentially a left-elbow
    circuit.append(
        cirq.X.on(clean_ancilla[0]).controlled_by(
            *control_qubits, control_values=control_values
        )
    )

    # Reverse loop because operators act starting from the right
    for mode in operator.modes[::-1]:
        for system_qubit in system[::-1][:mode]:
            circuit.append(cirq.Z.on(system_qubit).controlled_by(clean_ancilla[0]))
        circuit.append(cirq.X.on(system[-mode - 1]).controlled_by(clean_ancilla[0]))

    circuit.append(cirq.X.on(validation).controlled_by(clean_ancilla[0]))

    # This is essentially a right-elbow
    circuit.append(
        cirq.X.on(clean_ancilla[0]).controlled_by(
            *([validation] + index_register),
            control_values=[0] + index_register_control_values,
        )
    )

    return circuit
