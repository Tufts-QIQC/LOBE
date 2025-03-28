import cirq
import math
import numpy as np
from functools import partial
from .addition import add_classical_value
from .bosonic import (
    bosonic_product_block_encoding,
    _get_bosonic_rotation_angles,
    bosonic_product_plus_hc_block_encoding,
)
from .decompose import decompose_controls_left, decompose_controls_right
from .fermionic import (
    fermionic_product_block_encoding,
    _apply_fermionic_ladder_op,
    fermionic_plus_hc_block_encoding,
)
from .metrics import CircuitMetrics
from .multiplexed_rotations import get_decomposed_multiplexed_rotation_circuit
from ._utils import (
    get_bosonic_exponents,
    _apply_negative_identity,
    get_fermionic_operator_types,
)


def yukawa_term_block_encoding(
    system,
    block_encoding_ancillae,
    fermionic_indices,
    fermionic_operator_types,
    bosonic_indices,
    bosonic_exponents_list,
    sign=1,
    clean_ancillae=[],
    ctrls=([], []),
):
    """Add a block-encoding circuit for the 4 point pair term in the full Yukawa model

    NOTE: Term is expected to be in the form: $b_i b_j ... b_m a_k^\dagger a_l^\dagger ... a_t^\dagger + h.c.$.
        Expected ordering of fermionic indices is [m, ..., j, i].
        Expected ordering of bosonic indices is [t, ..., k, l].

    Args:
        system (lobe.system.System): The system object holding the system registers
        block_encoding_ancillae (List[cirq.LineQubit]): The block-encoding ancillae qubits
        active_indices (List[int]): A list of the modes upon which the ladder operators act. Assumed to be in order
            of which operators are applied (right to left).
        sign (int): Either 1 or -1 to indicate the sign of the term
        clean_ancillae (List[cirq.LineQubit]): A list of qubits that are promised to start and end in the 0-state.
        ctrls (Tuple(List[cirq.LineQubit], List[int])): A set of qubits and integers that correspond to
            the control qubits and values.

    Returns:
        - List of cirq operations representing the gates to be applied in the circuit
        - CircuitMetrics object representing cost of block-encoding circuit
    """
    assert len(ctrls[0]) == 1
    assert ctrls[1] == [1]
    gates = []
    block_encoding_metrics = CircuitMetrics()
    be_counter = 0

    for fermionic_index, operator_type in zip(
        fermionic_indices, fermionic_operator_types
    ):
        if operator_type == 1:
            gates.append(cirq.X.on(system.fermionic_modes[fermionic_index]))

    if sign == -1:
        gates += _apply_negative_identity(
            system.fermionic_modes[fermionic_indices[0]], ctrls=ctrls
        )

    temporary_qbool = system.fermionic_modes[fermionic_indices[0]]

    _gates, _metrics = decompose_controls_left(
        (ctrls[0] + [temporary_qbool], ctrls[1] + [1]), clean_ancillae[0]
    )
    gates += _gates
    block_encoding_metrics += _metrics

    for be_ancilla, bosonic_index, exponents in zip(
        block_encoding_ancillae, bosonic_indices, bosonic_exponents_list
    ):
        adder_gates, adder_metrics = add_classical_value(
            system.bosonic_modes[bosonic_index],
            exponents[0] - exponents[1],
            clean_ancillae=clean_ancillae[1:],
            ctrls=([clean_ancillae[0]], [1]),
        )
        gates += adder_gates
        block_encoding_metrics += adder_metrics

        rotation_angles = _get_bosonic_rotation_angles(
            system.maximum_occupation_number, exponents[0], exponents[1]
        )
        rotation_gates, rotation_metrics = get_decomposed_multiplexed_rotation_circuit(
            system.bosonic_modes[bosonic_index],
            be_ancilla,
            rotation_angles,
            clean_ancillae=clean_ancillae[1:],
            ctrls=ctrls,
        )
        gates += rotation_gates
        block_encoding_metrics += rotation_metrics
        be_counter += 1

    gates.append(
        cirq.X.on(clean_ancillae[0]).controlled_by(*ctrls[0], control_values=ctrls[1])
    )
    for be_ancilla, bosonic_index, exponents in zip(
        block_encoding_ancillae, bosonic_indices, bosonic_exponents_list
    ):
        adder_gates, adder_metrics = add_classical_value(
            system.bosonic_modes[bosonic_index],
            -(exponents[0] - exponents[1]),
            clean_ancillae=clean_ancillae[1:],
            ctrls=([clean_ancillae[0]], [1]),
        )
        gates += adder_gates
        block_encoding_metrics += adder_metrics

    _gates, _metrics = decompose_controls_right(
        (ctrls[0] + [temporary_qbool], ctrls[1] + [0]), clean_ancillae[0]
    )
    gates += _gates
    block_encoding_metrics += _metrics

    gates.append(
        cirq.X.on(temporary_qbool).controlled_by(
            system.fermionic_modes[fermionic_indices[1]]
        )
    )
    block_encoding_metrics.number_of_elbows += 1
    block_encoding_metrics.add_to_clean_ancillae_usage(1)
    gates.append(
        cirq.X.on(block_encoding_ancillae[be_counter]).controlled_by(
            *ctrls[0], temporary_qbool, control_values=ctrls[1] + [1]
        )
    )
    block_encoding_metrics.add_to_clean_ancillae_usage(-1)
    gates.append(
        cirq.X.on(temporary_qbool).controlled_by(
            system.fermionic_modes[fermionic_indices[1]]
        )
    )

    for fermionic_index, operator_type in zip(
        fermionic_indices, fermionic_operator_types
    ):
        if operator_type == 1:
            gates.append(cirq.X.on(system.fermionic_modes[fermionic_index]))

    # Potential sign flip
    number_of_swaps = math.comb(len(fermionic_indices), 2)
    if number_of_swaps % 2:
        sign_qubit = system.fermionic_modes[fermionic_indices[0]]
        if not fermionic_operator_types[0]:
            gates.append(cirq.Moment(cirq.X.on(sign_qubit)))
        gates.append(
            cirq.Moment(
                cirq.Z.on(sign_qubit).controlled_by(
                    *ctrls[0],
                    control_values=ctrls[1],
                )
            )
        )
        if not fermionic_operator_types[0]:
            gates.append(cirq.Moment(cirq.X.on(sign_qubit)))

    op_gates, op_metrics = _apply_fermionic_ladder_op(
        system, fermionic_indices[0], ctrls=ctrls
    )
    gates += op_gates
    block_encoding_metrics += op_metrics
    op_gates, op_metrics = _apply_fermionic_ladder_op(
        system, fermionic_indices[1], ctrls=ctrls
    )
    gates += op_gates
    block_encoding_metrics += op_metrics

    return gates, block_encoding_metrics


def _determine_block_encoding_function(
    group, system, block_encoding_ancillae, clean_ancillae
):
    term = group.to_list()[0].mode_order()
    fermionic_modes, fermionic_operator_types = get_fermionic_operator_types(term)
    bosonic_modes, bosonic_exponents_list = get_bosonic_exponents(term)
    if len(group.op_dict.keys()) == 2:

        if len(bosonic_modes) == 0:
            be_function = partial(
                fermionic_plus_hc_block_encoding,
                system=system,
                block_encoding_ancillae=block_encoding_ancillae[:1],
                active_indices=fermionic_modes[::-1],
                operator_types=fermionic_operator_types[::-1],
                sign=np.sign(term.coeff),
                clean_ancillae=clean_ancillae[::-1],
            )
        elif len(fermionic_modes) == 0:
            be_function = partial(
                bosonic_product_plus_hc_block_encoding,
                system=system,
                block_encoding_ancillae=block_encoding_ancillae,
                active_indices=bosonic_modes,
                exponents_list=bosonic_exponents_list,
                sign=np.sign(term.coeff),
                clean_ancillae=clean_ancillae[::-1],
            )
        else:
            be_function = partial(
                yukawa_term_block_encoding,
                system=system,
                block_encoding_ancillae=block_encoding_ancillae,
                fermionic_indices=fermionic_modes[::-1],
                fermionic_operator_types=fermionic_operator_types[::-1],
                bosonic_indices=bosonic_modes,
                bosonic_exponents_list=bosonic_exponents_list,
                sign=np.sign(term.coeff),
                clean_ancillae=clean_ancillae[::-1],
            )
        power = sum([sum(exponents) for exponents in bosonic_exponents_list])
        return be_function, np.sqrt(system.maximum_occupation_number) ** power
    else:

        def _helper(ctrls=([], [])):
            be_counter = 0
            _gates = []
            _metrics = CircuitMetrics()

            if np.isclose(np.sign(term.coeff), -1):
                _gates += _apply_negative_identity(
                    system.fermionic_modes[0], ctrls=ctrls
                )

            if len(fermionic_modes) > 0:
                __gates, __metrics = fermionic_product_block_encoding(
                    system=system,
                    block_encoding_ancillae=[block_encoding_ancillae[be_counter]],
                    active_indices=fermionic_modes[::-1],
                    operator_types=fermionic_operator_types[::-1],
                    clean_ancillae=clean_ancillae[::-1],
                    ctrls=ctrls,
                )
                _gates += __gates
                _metrics += __metrics
                be_counter += 1

            if len(bosonic_modes) > 0:
                __gates, __metrics = bosonic_product_block_encoding(
                    system=system,
                    block_encoding_ancillae=block_encoding_ancillae[
                        be_counter : be_counter + len(bosonic_modes)
                    ],
                    active_indices=bosonic_modes,
                    exponents_list=bosonic_exponents_list,
                    clean_ancillae=clean_ancillae[::-1],
                    ctrls=ctrls,
                )
                _gates += __gates
                _metrics += __metrics
                be_counter += len(bosonic_modes)

            return _gates, _metrics

        return _helper, np.sqrt(system.maximum_occupation_number) ** (
            sum([sum(exponents) for exponents in bosonic_exponents_list])
        )
