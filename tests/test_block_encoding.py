from openparticle import ParticleOperator
import numpy as np
import cirq
from src.lobe.system import System
from src.lobe.block_encoding import add_lobe_oracle


def test():
    terms = [
        ParticleOperator("b1^ a0 b0"),
        ParticleOperator("a1^ b0^ a0"),
        ParticleOperator("a0^ b1 a1"),
    ]

    coefficients = np.random.uniform(-1, 1, size=len(terms))

    maximum_occupation_number = 3
    number_of_modes = max([mode for term in terms for mode in term.modes]) + 1
    number_of_index_qubits = max(int(np.ceil(np.log2(len(terms)))), 1)
    number_of_ancillae = 4
    circuit = cirq.Circuit()
    validation = cirq.LineQubit(0)
    clean_ancillae = [cirq.LineQubit(i + 1) for i in range(number_of_ancillae)]
    rotation_qubits = [cirq.LineQubit(i + 1 + number_of_ancillae) for i in range(3)]
    index_register = [
        cirq.LineQubit(i + 1 + number_of_ancillae + 3)
        for i in range(number_of_index_qubits)
    ]

    system = System(
        number_of_modes=number_of_modes,
        maximum_occupation_number=maximum_occupation_number,
        number_of_used_qubits=1 + number_of_ancillae + 3 + number_of_index_qubits,
        has_fermions=True,
        has_bosons=True,
    )
    normalization_factor = max(coefficients)
    normalized_coefficients = coefficients / (
        normalization_factor * maximum_occupation_number
    )

    circuit += add_lobe_oracle(
        terms,
        validation,
        index_register,
        system,
        rotation_qubits,
        clean_ancillae,
        perform_coefficient_oracle=True,
    )

    # with open("lobe.svg", "w") as f:
    #     f.write(SVGCircuit(circuit)._repr_svg_())
    print(len(circuit.all_qubits()))
    upper_left_block = circuit.unitary(dtype=np.complex64)[
        : 1 << system.number_of_system_qubits, : 1 << system.number_of_system_qubits
    ]
