from openparticle import ParticleOperator
from openparticle.utils import generate_matrix, get_fock_basis
import numpy as np
import cirq
from symmer import PauliwordOp
from symmer.operators.utils import symplectic_to_string

import sys, os

from .asp import get_target_state, add_prepare_circuit


def seperate_real_imag(Pop: PauliwordOp) -> PauliwordOp:
    """
    seperate the real and imaginary part of a PauliwordOp into seperate terms!
    This is useful for block encodings when ops have real and imag coeffs.

    IMPORTANT: .cleanup() should NOT be used on the output as this will combine the coefficients again.
    operations on the output will perform a cleanup (so addition, multiplication etc... should not be used)

    Args:
        Pop (PauliwordOp): op to split into real and imag parts. Input is assumed to be cleaned up.
    Returns
        A PauliwordOp that has real and imaginary coefficients on seperate Pauli terms

    """

    op_real = Pop[np.abs(Pop.coeff_vec.real) > 0]
    op_real.coeff_vec = op_real.coeff_vec.real
    op_imag = Pop[np.abs(Pop.coeff_vec.imag) > 0]
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

    def __init__(self, operator: ParticleOperator, max_bose_occ: int):

        paulis = operator.to_paulis(max_bose_occ=max_bose_occ)

        self.paulis = seperate_real_imag(paulis)

        self.coeffs = self.paulis.coeff_vec

        self.prep_coeffs, self.one_norm = self.get_prep_vector(self.coeffs)

        self.target_state = get_target_state(self.coeffs)
        self.number_of_index_qubits = max(int(np.ceil(np.log2(self.paulis.n_terms))), 1)
        self.index_register = [
            cirq.LineQubit(i) for i in range(self.number_of_index_qubits)
        ]
        self.system_register = [
            cirq.LineQubit(self.number_of_index_qubits + i)
            for i in range(self.paulis.n_qubits)
        ]
        self.number_of_system_qubits = len(self.system_register)

    @staticmethod
    def get_prep_vector(coeff_vector):
        pre_process_coeffs = np.real(coeff_vector) + np.imag(
            coeff_vector
        )  # 1j coefficient becomes 1

        one_norm = np.linalg.norm(pre_process_coeffs, ord=1)

        pre_coeffs = np.sqrt(np.abs(pre_process_coeffs) / one_norm)

        return pre_coeffs, one_norm

    def get_circuit(self):
        self.circuit = cirq.Circuit()

        self.circuit += add_prepare_circuit(
            self.index_register, target_state=self.target_state
        )

        self.circuit += self.add_select_oracle(
            self.index_register, self.paulis, system_register=self.system_register
        )

        self.circuit += add_prepare_circuit(
            self.index_register, target_state=self.target_state, dagger=True
        )

        return self.circuit

    @classmethod
    def add_select_oracle(cls, index_register, paulis, system_register):
        gates = []

        n_index_qubits = len(index_register)
        terms = [symplectic_to_string(row) for row in paulis.symp_matrix]
        n_terms = len(terms)
        coeffs = paulis.coeff_vec  # list(paulis.to_dictionary.values())
        n_system_qubits = paulis.n_qubits

        for i in range(n_terms):
            ctrl = f"{i:0{n_index_qubits}b}"
            for n in range(n_system_qubits):
                term = terms[i][n]
                if n == 0:
                    # Check for +-1j or -1 coeff only needed for one term in tensor product
                    key = (term, cls.map_complex_to_key(coeffs[i]))
                    gate = GATE_SELECTION[key]
                else:
                    gate = GATE_SELECTION[(term, 1)]
                for mapped_gates in gate:
                    gates.append(
                        mapped_gates.on(system_register[n]).controlled_by(
                            *index_register, control_values=[int(char) for char in ctrl]
                        )
                    )  # TODO: remove controlled outer gates in -X = ZXZ, -Y = ZYZ, -Z = XZX
        return gates

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
