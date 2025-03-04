import cirq
import numpy as np
from functools import partial
from symmer import PauliwordOp
from symmer.operators.utils import symplectic_to_string
from .asp import get_target_state, add_prepare_circuit
from .index import index_over_terms
from .metrics import CircuitMetrics


def estimate_pauli_lcu_rescaling_factor_and_number_of_be_ancillae(
    system, operator, zero_threshold=1e-6
):
    """Estimate rescaling factor and number of block encoding ancillae for Pauli LCU

    Args:
        system (lobe.system.System): The system object holding the system registers
        operator (openparticle.ParticleOperator): The operator to transform into the LCU of Paulis
        zero_threshold (float): The cutoff value for the coefficients in the LCU

    Returns:
        - Float representing rescaling factor
    """
    paulis = operator.to_paulis(
        max_fermionic_mode=operator.max_fermionic_mode,
        max_antifermionic_mode=operator.max_antifermionic_mode,
        max_bosonic_mode=operator.max_bosonic_mode,
        max_bosonic_occupancy=system.maximum_occupation_number,
        zero_threshold=zero_threshold,
    )
    paulis = seperate_real_imag(paulis, zero_threshold=zero_threshold)
    _, rescaling_factor = _get_prep_vector(paulis.coeff_vec)
    return rescaling_factor, max(int(np.ceil(np.log2(paulis.n_terms))), 1)


def pauli_lcu_block_encoding(
    system,
    block_encoding_ancillae,
    system_register,
    paulis,
    zero_threshold=1e-6,
    clean_ancillae=[],
    ctrls=([], []),
):
    """Obtain operations for a block-encoding circuit for an LCU of pauli operators

    Args:
        system (lobe.system.System): The system object holding the system registers
        block_encoding_ancillae (List[cirq.LineQubit]): A list of ancillae used to block-encode the LCU
        operator (openparticle.ParticleOperator): The operator to transform into the LCU of Paulis
        zero_threshold (float): The cutoff value for the coefficients in the LCU
        clean_ancillae (List[cirq.LineQubit]): A list of qubits that are promised to start and end in the 0-state.
        ctrls (Tuple(List[cirq.LineQubit], List[int])): A set of qubits and integers that correspond to
            the control qubits and values.

    Returns:
        - List of cirq operations representing the gates to be applied in the circuit
        - CircuitMetrics object representing cost of block-encoding circuit
    """
    paulis = seperate_real_imag(paulis, zero_threshold=zero_threshold)

    # Check number of block-encoding ancillae is correct
    assert len(block_encoding_ancillae) == max(int(np.ceil(np.log2(paulis.n_terms))), 1)

    prep_state = get_target_state(paulis.coeff_vec)

    gates = []
    circuit_metrics = CircuitMetrics()

    _gates, _metrics = add_prepare_circuit(
        block_encoding_ancillae,
        target_state=prep_state,
        clean_ancillae=clean_ancillae,
    )
    gates += _gates
    circuit_metrics += _metrics

    _gates, _metrics = _select_paulis(
        block_encoding_ancillae,
        paulis,
        system_register=system_register,
        clean_ancillae=clean_ancillae,
        ctrls=ctrls,
    )
    gates += _gates
    circuit_metrics += _metrics

    _gates, _metrics = add_prepare_circuit(
        block_encoding_ancillae,
        target_state=prep_state,
        dagger=True,
        clean_ancillae=clean_ancillae,
    )
    gates += _gates
    circuit_metrics += _metrics

    return gates, circuit_metrics


def seperate_real_imag(Pop: PauliwordOp, zero_threshold: float = 1e-15) -> PauliwordOp:
    """
    seperate the real and imaginary part of a PauliwordOp into seperate terms!
    This is useful for block encodings when ops have real and imag coeffs.

    IMPORTANT: .cleanup() should NOT be used on the output as this will combine the coefficients again.
    operations on the output will perform a cleanup (so addition, multiplication etc... should not be used)

    Args:
        Pop (PauliwordOp): op to split into real and imag parts. Input is assumed to be cleaned up.
        zero_threshold (float): The cutoff on the magnitude of the coefficients. Terms will smaller coefficients will
            be removed.
    Returns
        A PauliwordOp that has real and imaginary coefficients on seperate Pauli terms

    """

    op_real = Pop[np.abs(Pop.coeff_vec.real) > zero_threshold]
    op_real.coeff_vec = op_real.coeff_vec.real
    op_imag = Pop[np.abs(Pop.coeff_vec.imag) > zero_threshold]
    op_imag.coeff_vec = op_imag.coeff_vec.imag * 1j

    return PauliwordOp(
        np.vstack((op_real.symp_matrix, op_imag.symp_matrix)),
        np.hstack((op_real.coeff_vec, op_imag.coeff_vec)),
    )


GATE_SELECTION = {
    ("Z", 1): [cirq.Z],
    ("Z", 1j): [cirq.rz(-np.pi)],
    ("Z", -1j): [cirq.rz(np.pi)],
    ("Z", -1): [cirq.X, cirq.Z, cirq.X],
    ("X", 1): [cirq.X],
    ("X", 1j): [cirq.rx(-np.pi)],
    ("X", -1j): [cirq.rx(np.pi)],
    ("X", -1): [cirq.Z, cirq.X, cirq.Z],
    ("Y", 1): [cirq.Y],
    ("Y", 1j): [cirq.ry(-np.pi)],
    ("Y", -1j): [cirq.ry(np.pi)],
    ("Y", -1): [cirq.Z, cirq.Y, cirq.Z],
    ("I", 1): [cirq.I],
    ("I", 1j): [cirq.rz(-np.pi), cirq.Z],
    ("I", -1j): [cirq.rz(np.pi), cirq.Z],
    ("I", -1): [cirq.rz(2 * np.pi)],
}


def _map_complex_to_key(complex_number):
    if np.real(complex_number) > 0:
        return 1
    elif np.real(complex_number) < 0:
        return -1
    elif np.imag(complex_number) > 0:
        return 1j
    elif np.imag(complex_number) < 0:
        return -1j


def _get_prep_vector(coeff_vector):
    pre_process_coeffs = np.real(coeff_vector) + np.imag(
        coeff_vector
    )  # 1j coefficient becomes 1

    one_norm = np.linalg.norm(pre_process_coeffs, ord=1)

    pre_coeffs = np.sqrt(np.abs(pre_process_coeffs) / one_norm)

    return pre_coeffs, one_norm


def _select_paulis(
    index_register, paulis, system_register, clean_ancillae=[], ctrls=([], [])
):
    gates = []

    terms = [symplectic_to_string(row) for row in paulis.symp_matrix]
    coeffs = paulis.coeff_vec  # list(paulis.to_dictionary.values())
    n_system_qubits = paulis.n_qubits

    def _apply_term(term, coeff, ctrls=([], [])):
        gates = []
        for n in range(n_system_qubits):
            operator = term[n]
            if n == 0:
                # Check for +-1j or -1 coeff only needed for one term in tensor product
                key = (operator, _map_complex_to_key(coeff))
                operations = GATE_SELECTION[key]
            else:
                operations = GATE_SELECTION[(operator, 1)]

            for operation in operations:
                gates.append(
                    operation.on(system_register[n]).controlled_by(
                        *ctrls[0], control_values=ctrls[1]
                    )
                )  # TODO: remove controlled outer gates in -X = ZXZ, -Y = ZYZ, -Z = XZX

        return gates, CircuitMetrics()

    _gates, metrics = index_over_terms(
        index_register,
        [
            partial(_apply_term, term=term, coeff=coeff)
            for term, coeff in zip(terms, coeffs)
        ],
        clean_ancillae=clean_ancillae,
        ctrls=ctrls,
    )
    gates += _gates
    return gates, metrics
