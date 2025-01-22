from openparticle import ParticleOperator
import numpy as np
import cirq
from symmer import PauliwordOp
from symmer.operators.utils import symplectic_to_string
from .metrics import CircuitMetrics
from .asp import get_target_state, add_prepare_circuit
from .index import index_over_terms
from functools import partial


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


class LCU:

    def __init__(
        self,
        operator: ParticleOperator,
        max_bosonic_occupancy: int,
        zero_threshold: float = 1e-15,
    ):
        self.operator = operator
        self.zero_threshold = zero_threshold
        paulis = operator.to_paulis(
            max_fermionic_mode=operator.max_fermionic_mode,
            max_antifermionic_mode=operator.max_antifermionic_mode,
            max_bosonic_mode=operator.max_bosonic_mode,
            max_bosonic_occupancy=max_bosonic_occupancy,
            zero_threshold=zero_threshold,
        )

        self.paulis = seperate_real_imag(paulis, zero_threshold=zero_threshold)

        self.coeffs = self.paulis.coeff_vec

        self.prep_coeffs, self.one_norm = self.get_prep_vector(self.coeffs)

        self.target_state = get_target_state(self.coeffs)
        self.number_of_index_qubits = max(int(np.ceil(np.log2(self.paulis.n_terms))), 1)
        self.index_register = [
            cirq.LineQubit(i) for i in range(self.number_of_index_qubits)
        ]
        self.system_register = [
            cirq.LineQubit(1000 + i + self.number_of_index_qubits)
            for i in range(self.paulis.n_qubits)
        ]
        self.number_of_system_qubits = len(self.system_register)
        self.clean_ancillae = [cirq.LineQubit(-1 - i) for i in range(100)]
        self.circuit_metrics = CircuitMetrics()

    @staticmethod
    def get_prep_vector(coeff_vector):
        pre_process_coeffs = np.real(coeff_vector) + np.imag(
            coeff_vector
        )  # 1j coefficient becomes 1

        one_norm = np.linalg.norm(pre_process_coeffs, ord=1)

        pre_coeffs = np.sqrt(np.abs(pre_process_coeffs) / one_norm)

        return pre_coeffs, one_norm

    def get_circuit(self, ctrls=([], [])):
        self.circuit = cirq.Circuit()
        self.circuit_metrics = CircuitMetrics()

        _gates, _metrics = add_prepare_circuit(
            self.index_register,
            target_state=self.target_state,
            clean_ancillae=self.clean_ancillae,
        )
        self.circuit.append(_gates)
        self.circuit_metrics += _metrics

        _gates, _metrics = self.add_select_oracle(
            self.index_register,
            self.paulis,
            system_register=self.system_register,
            clean_ancillae=self.clean_ancillae,
            ctrls=ctrls,
        )
        self.circuit.append(_gates)
        self.circuit_metrics += _metrics

        _gates, _metrics = add_prepare_circuit(
            self.index_register,
            target_state=self.target_state,
            dagger=True,
            clean_ancillae=self.clean_ancillae,
        )
        self.circuit.append(_gates)
        self.circuit_metrics += _metrics

        return self.circuit

    @classmethod
    def add_select_oracle(
        cls, index_register, paulis, system_register, clean_ancillae=[], ctrls=([], [])
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
                    key = (operator, cls.map_complex_to_key(coeff))
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

    @property
    def unitary(self):
        circuit = self.get_circuit()
        upper_left_block = circuit.unitary(dtype=complex)[
            : 1 << self.number_of_system_qubits, : 1 << self.number_of_system_qubits
        ]
        return upper_left_block * self.one_norm

    @staticmethod
    def map_complex_to_key(complex_number):
        if np.real(complex_number) > 0:
            return 1
        elif np.real(complex_number) < 0:
            return -1
        elif np.imag(complex_number) > 0:
            return 1j
        elif np.imag(complex_number) < 0:
            return -1j
