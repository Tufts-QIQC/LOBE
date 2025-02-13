import cirq
import numpy as np


class System:

    def __init__(
        self,
        maximum_occupation_number=0,
        number_of_used_qubits=0,
        number_of_fermionic_modes=0,
        number_of_bosonic_modes=0,
    ):
        self.maximum_occupation_number = maximum_occupation_number
        self.number_of_system_qubits = 0
        self.number_of_fermionic_modes = number_of_fermionic_modes
        self.number_of_bosonic_modes = number_of_bosonic_modes
        self.bosonic_modes = []

        self.fermionic_modes = [
            cirq.LineQubit(i + number_of_used_qubits)
            for i in range(self.number_of_fermionic_modes)
        ]
        self.number_of_system_qubits += self.number_of_fermionic_modes

        size_of_bosonic_register = max(
            int(np.ceil(np.log2(self.maximum_occupation_number + 1))), 1
        )

        self.bosonic_modes = [
            [
                cirq.LineQubit(
                    number_of_used_qubits
                    + self.number_of_system_qubits
                    + (i + (j * size_of_bosonic_register))
                )
                for i in range(size_of_bosonic_register)
            ]
            for j in range(self.number_of_bosonic_modes)
        ]
        self.number_of_system_qubits += (
            self.number_of_bosonic_modes * size_of_bosonic_register
        )
