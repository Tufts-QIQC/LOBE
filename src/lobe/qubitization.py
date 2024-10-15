from .block_encoding import add_lobe_oracle
from .asp import add_prepare_circuit, get_target_state
import cirq


def _add_reflection(target_state, index_register, ctrls=([], [])):
    gates = []
    gates += add_prepare_circuit(index_register, target_state=target_state, dagger=True)
    if len(ctrls[0]) == 0:
        gates.append(cirq.X.on(index_register[0]))
        gates.append(
            cirq.Z.on(index_register[0]).controlled_by(
                *index_register[1:],
                control_values=[0] * (len(index_register) - 1),
            )
        )
        gates.append(cirq.X.on(index_register[0]))
    elif len(ctrls[0]) == 1:
        if ctrls[1] == 0:
            gates.append(cirq.X.on(ctrls[0][0]))
        gates.append(
            cirq.Z.on_each(*ctrls[0]).controlled_by(
                *index_register[1:],
                control_values=[0] * (len(index_register) - 1),
            )
        )
        if ctrls[1] == 0:
            gates.append(cirq.X.on(ctrls[0][0]))
    else:
        raise RuntimeError("Length of quantum controls must be zero or one.")
    gates += add_prepare_circuit(index_register, target_state=target_state)
    return gates


def add_qubitized_walk_operator(
    terms,
    coefficients,
    validation,
    clean_ancillae_register,
    rotation_register,
    index_register,
    system,
    ctrls=([], []),
):
    target_state = get_target_state(coefficients)

    gates = []
    gates += add_lobe_oracle(
        terms,
        validation,
        index_register,
        system,
        rotation_register,
        clean_ancillae_register,
        perform_coefficient_oracle=False,
        decompose=True,
        ctrls=ctrls,
    )

    gates += _add_reflection(target_state, index_register, ctrls)

    gates.append(cirq.X.on(validation))
    gates.append(cirq.Y.on(validation))
    gates.append(cirq.X.on(validation))
    gates.append(cirq.Y.on(validation))

    return gates
