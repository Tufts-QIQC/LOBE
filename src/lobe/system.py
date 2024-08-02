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
        self.has_fermions = has_fermions
        self.has_antifermions = has_antifermions
        self.has_bosons = has_bosons
        self.number_of_system_qubits = 0
        self.fermionic_register = []
        self.antifermionic_register = []
        self.bosonic_occupation_register = []

        if self.has_fermions:
            self.fermionic_register = [
                cirq.LineQubit(i + number_of_used_qubits)
                for i in range(self.number_of_modes)
            ][::-1]
            self.number_of_system_qubits += self.number_of_modes

        if self.has_antifermions:
            self.antifermionic_register = [
                cirq.LineQubit(i + number_of_used_qubits + self.number_of_system_qubits)
                for i in range(self.number_of_modes)
            ][::-1]
            self.number_of_system_qubits += self.number_of_modes

        if self.has_bosons:
            number_of_occupation_qubits = max(
                int(np.ceil(np.log2(self.maximum_occupation_number + 1))), 1
            )

            self.bosonic_system = [
                [
                    cirq.LineQubit(
                        number_of_used_qubits
                        + self.number_of_system_qubits
                        + (i + (j * number_of_occupation_qubits))
                    )
                    for i in range(number_of_occupation_qubits)
                ]
                for j in range(self.number_of_modes)
            ]
            # reverse order so that occupation_0 is at index 0
            # self.bosonic_system = self.bosonic_system[::-1]
            self.number_of_system_qubits += (
                self.number_of_modes * number_of_occupation_qubits
            )
