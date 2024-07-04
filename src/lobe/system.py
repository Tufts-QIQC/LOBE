import cirq
import numpy as np


class System:

    def __init__(
        self,
        number_of_modes=0,
        maximum_occupation_number=0,
        number_of_used_qubits=0,
        has_fermions=False,
        has_antifermions=False,
        has_bosons=False,
    ):
        self.number_of_modes = number_of_modes
        self.maximum_occupation_number = maximum_occupation_number
        self.number_of_used_qubits = number_of_used_qubits
        self.has_fermions = has_fermions
        self.has_antifermions = has_antifermions
        self.has_bosons = has_bosons
        self.number_of_system_qubits = 0
        number_of_mode_qubits = int(np.ceil(np.log2(number_of_modes)))
        self.fermionic_register = []
        self.antifermionic_register = []
        self.bosonic_mode_register = []
        self.bosonic_occupation_register = []

        if self.has_fermions:
            self.fermionic_register = [
                cirq.LineQubit(i + number_of_used_qubits)
                for i in range(number_of_modes)
            ]
            self.number_of_system_qubits += number_of_modes

        if self.has_antifermions:
            self.antifermionic_register = [
                cirq.LineQubit(i + number_of_used_qubits + self.number_of_system_qubits)
                for i in range(number_of_modes)
            ]
            self.number_of_system_qubits += number_of_modes

        if self.has_bosons:
            number_of_occupation_qubits = int(
                np.ceil(np.log2(maximum_occupation_number))
            )
            self.bosonic_mode_register = [
                cirq.LineQubit(i + number_of_used_qubits + self.number_of_system_qubits)
                for i in range(number_of_mode_qubits)
            ]
            self.number_of_system_qubits += number_of_mode_qubits

            self.bosonic_occupation_register = [
                cirq.LineQubit(i + number_of_used_qubits + self.number_of_system_qubits)
                for i in range(number_of_occupation_qubits)
            ]
            self.number_of_system_qubits += number_of_occupation_qubits
