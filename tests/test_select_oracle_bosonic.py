import pytest
import numpy as np
import cirq

# from src.lobe.select_oracle import add_select_oracle
from src.lobe.block_encoding import add_lobe_oracle
from src.lobe.system import System
import copy
from openparticle import ParticleOperator

# H = a^\dagger_0 a_0 +  a^\dagger_1 a_1 + a^\dagger_0 a_1 +  a^\dagger_1 a_0
# validation, bosonic_rotation, clean_ancillae, index, occupancy_1, occupancy_0
TOY_BOSONIC_HAMILTONIAN_SELECT_STATE_MAP = {
    "1" + "00" + "00" + "00" + "00" + "00": "1" + "00" + "00" + "00" + "00" + "00",
    "1" + "00" + "00" + "00" + "00" + "01": "0" + "00" + "00" + "00" + "00" + "01",
    "1" + "00" + "00" + "00" + "00" + "10": "0" + "00" + "00" + "00" + "00" + "10",
    "1" + "00" + "00" + "00" + "00" + "11": "0" + "00" + "00" + "00" + "00" + "11",
    "1" + "00" + "00" + "00" + "01" + "00": "1" + "00" + "00" + "00" + "01" + "00",
    "1" + "00" + "00" + "00" + "01" + "01": "0" + "00" + "00" + "00" + "01" + "01",
    "1" + "00" + "00" + "00" + "01" + "10": "0" + "00" + "00" + "00" + "01" + "10",
    "1" + "00" + "00" + "00" + "01" + "11": "0" + "00" + "00" + "00" + "01" + "11",
    "1" + "00" + "00" + "00" + "10" + "00": "1" + "00" + "00" + "00" + "10" + "00",
    "1" + "00" + "00" + "00" + "10" + "01": "0" + "00" + "00" + "00" + "10" + "01",
    "1" + "00" + "00" + "00" + "10" + "10": "0" + "00" + "00" + "00" + "10" + "10",
    "1" + "00" + "00" + "00" + "10" + "11": "0" + "00" + "00" + "00" + "10" + "11",
    "1" + "00" + "00" + "00" + "11" + "00": "1" + "00" + "00" + "00" + "11" + "00",
    "1" + "00" + "00" + "00" + "11" + "01": "0" + "00" + "00" + "00" + "11" + "01",
    "1" + "00" + "00" + "00" + "11" + "10": "0" + "00" + "00" + "00" + "11" + "10",
    "1" + "00" + "00" + "00" + "11" + "11": "0" + "00" + "00" + "00" + "11" + "11",
    "1" + "00" + "00" + "01" + "00" + "00": "1" + "00" + "00" + "01" + "00" + "00",
    "1" + "00" + "00" + "01" + "00" + "01": "1" + "00" + "00" + "01" + "00" + "01",
    "1" + "00" + "00" + "01" + "00" + "10": "1" + "00" + "00" + "01" + "00" + "10",
    "1" + "00" + "00" + "01" + "00" + "11": "1" + "00" + "00" + "01" + "00" + "11",
    "1" + "00" + "00" + "01" + "01" + "00": "0" + "00" + "00" + "01" + "01" + "00",
    "1" + "00" + "00" + "01" + "01" + "01": "0" + "00" + "00" + "01" + "01" + "01",
    "1" + "00" + "00" + "01" + "01" + "10": "0" + "00" + "00" + "01" + "01" + "10",
    "1" + "00" + "00" + "01" + "01" + "11": "0" + "00" + "00" + "01" + "01" + "11",
    "1" + "00" + "00" + "01" + "10" + "00": "0" + "00" + "00" + "01" + "10" + "00",
    "1" + "00" + "00" + "01" + "10" + "01": "0" + "00" + "00" + "01" + "10" + "01",
    "1" + "00" + "00" + "01" + "10" + "10": "0" + "00" + "00" + "01" + "10" + "10",
    "1" + "00" + "00" + "01" + "10" + "11": "0" + "00" + "00" + "01" + "10" + "11",
    "1" + "00" + "00" + "01" + "11" + "00": "0" + "00" + "00" + "01" + "11" + "00",
    "1" + "00" + "00" + "01" + "11" + "01": "0" + "00" + "00" + "01" + "11" + "01",
    "1" + "00" + "00" + "01" + "11" + "10": "0" + "00" + "00" + "01" + "11" + "10",
    "1" + "00" + "00" + "01" + "11" + "11": "0" + "00" + "00" + "01" + "11" + "11",
    "1" + "00" + "00" + "10" + "00" + "00": "1" + "00" + "00" + "10" + "00" + "00",
    "1" + "00" + "00" + "10" + "00" + "01": "1" + "00" + "00" + "10" + "00" + "01",
    "1" + "00" + "00" + "10" + "00" + "10": "1" + "00" + "00" + "10" + "00" + "10",
    "1" + "00" + "00" + "10" + "00" + "11": "1" + "00" + "00" + "10" + "00" + "11",
    "1" + "00" + "00" + "10" + "01" + "00": "0" + "00" + "00" + "10" + "00" + "01",
    "1" + "00" + "00" + "10" + "01" + "01": "0" + "00" + "00" + "10" + "00" + "10",
    "1" + "00" + "00" + "10" + "01" + "10": "0" + "00" + "00" + "10" + "00" + "11",
    "1" + "00" + "00" + "10" + "01" + "11": "1" + "00" + "00" + "10" + "01" + "11",
    "1" + "00" + "00" + "10" + "10" + "00": "0" + "00" + "00" + "10" + "01" + "01",
    "1" + "00" + "00" + "10" + "10" + "01": "0" + "00" + "00" + "10" + "01" + "10",
    "1" + "00" + "00" + "10" + "10" + "10": "0" + "00" + "00" + "10" + "01" + "11",
    "1" + "00" + "00" + "10" + "10" + "11": "1" + "00" + "00" + "10" + "10" + "11",
    "1" + "00" + "00" + "10" + "11" + "00": "0" + "00" + "00" + "10" + "10" + "01",
    "1" + "00" + "00" + "10" + "11" + "01": "0" + "00" + "00" + "10" + "10" + "10",
    "1" + "00" + "00" + "10" + "11" + "10": "0" + "00" + "00" + "10" + "10" + "11",
    "1" + "00" + "00" + "10" + "11" + "11": "1" + "00" + "00" + "10" + "11" + "11",
    "1" + "00" + "00" + "11" + "00" + "00": "1" + "00" + "00" + "11" + "00" + "00",
    "1" + "00" + "00" + "11" + "00" + "01": "0" + "00" + "00" + "11" + "01" + "00",
    "1" + "00" + "00" + "11" + "00" + "10": "0" + "00" + "00" + "11" + "01" + "01",
    "1" + "00" + "00" + "11" + "00" + "11": "0" + "00" + "00" + "11" + "01" + "10",
    "1" + "00" + "00" + "11" + "01" + "00": "1" + "00" + "00" + "11" + "01" + "00",
    "1" + "00" + "00" + "11" + "01" + "01": "0" + "00" + "00" + "11" + "10" + "00",
    "1" + "00" + "00" + "11" + "01" + "10": "0" + "00" + "00" + "11" + "10" + "01",
    "1" + "00" + "00" + "11" + "01" + "11": "0" + "00" + "00" + "11" + "10" + "10",
    "1" + "00" + "00" + "11" + "10" + "00": "1" + "00" + "00" + "11" + "10" + "00",
    "1" + "00" + "00" + "11" + "10" + "01": "0" + "00" + "00" + "11" + "11" + "00",
    "1" + "00" + "00" + "11" + "10" + "10": "0" + "00" + "00" + "11" + "11" + "01",
    "1" + "00" + "00" + "11" + "10" + "11": "0" + "00" + "00" + "11" + "11" + "10",
    "1" + "00" + "00" + "11" + "11" + "00": "1" + "00" + "00" + "11" + "11" + "00",
    "1" + "00" + "00" + "11" + "11" + "01": "1" + "00" + "00" + "11" + "11" + "01",
    "1" + "00" + "00" + "11" + "11" + "10": "1" + "00" + "00" + "11" + "11" + "10",
    "1" + "00" + "00" + "11" + "11" + "11": "1" + "00" + "00" + "11" + "11" + "11",
}


# def get_bosonic_select_oracle_test_inputs():
#     simulator = cirq.Simulator(dtype=np.complex128)
#     number_of_index_qubits = 2
#     operators = (
#         ParticleOperator("a0^ a0")
#         + ParticleOperator("a1^ a1")
#         + ParticleOperator("a0^ a1")
#         + ParticleOperator("a1^ a0")
#     ).to_list()

#     circuit = cirq.Circuit()
#     validation = cirq.LineQubit(0)
#     bosonic_rotation_register = [cirq.LineQubit(i + 1) for i in range(2)]
#     clean_ancillae = [cirq.LineQubit(i + 3) for i in range(2)]
#     index = [cirq.LineQubit(i + 5) for i in range(2)]
#     system = System(
#         number_of_modes=2,
#         maximum_occupation_number=4,
#         number_of_used_qubits=7,
#         has_bosons=True,
#     )

#     circuit.append(cirq.I.on_each(*bosonic_rotation_register))
#     circuit.append(cirq.I.on_each(*clean_ancillae))

#     circuit = add_select_oracle(
#         circuit,
#         validation,
#         index,
#         system,
#         operators,
#         bosonic_rotation_register,
#         clean_ancillae,
#     )

#     initial_state_of_validation = np.zeros(2)
#     initial_state_of_validation[1] = 1  # |1>
#     initial_state_of_clean_ancillae = np.zeros(1 << 4)
#     initial_state_of_clean_ancillae[0] = 1  # |0>
#     initial_state_of_validation_and_control = np.kron(
#         initial_state_of_validation, initial_state_of_clean_ancillae
#     )  # |1> tensor |0>

#     intitial_state_of_index = (
#         np.random.uniform(-1, 1, 1 << number_of_index_qubits)
#         + np.random.uniform(-1, 1, 1 << number_of_index_qubits) * 1j
#     )
#     intitial_state_of_index /= np.linalg.norm(intitial_state_of_index)
#     intitial_state_of_val_control_index = np.kron(
#         initial_state_of_validation_and_control, intitial_state_of_index
#     )

#     return simulator, circuit, intitial_state_of_val_control_index, system


# @pytest.mark.parametrize("index_bitstring", ["00", "01", "10", "11"])
# @pytest.mark.parametrize("occupancy_1", ["00", "01", "10", "11"])
# @pytest.mark.parametrize("occupancy_0", ["00", "01", "10", "11"])
# def test_select_oracle_on_basis_state_for_toy_bosonic_hamiltonian(
#     index_bitstring, occupancy_1, occupancy_0
# ):
#     simulator, circuit, intitial_state_of_val_control_index, system = (
#         get_bosonic_select_oracle_test_inputs()
#     )

#     initial_state_of_occupancy_1 = np.zeros(1 << len(occupancy_1))
#     initial_state_of_occupancy_1[int(occupancy_1, 2)] = 1
#     initial_state_of_occupancy_0 = np.zeros(1 << len(occupancy_0))
#     initial_state_of_occupancy_0[int(occupancy_0, 2)] = 1
#     initial_state = np.kron(
#         intitial_state_of_val_control_index,
#         np.kron(initial_state_of_occupancy_1, initial_state_of_occupancy_0),
#     )

#     wavefunction = simulator.simulate(
#         circuit, initial_state=initial_state
#     ).final_state_vector

#     initial_bitstring = "1" + "00" + "00" + index_bitstring + occupancy_1 + occupancy_0
#     assert initial_state[int(initial_bitstring, 2)] != 0

#     if TOY_BOSONIC_HAMILTONIAN_SELECT_STATE_MAP[initial_bitstring][0] == "0":
#         omega = 3
#         expected_coefficient = 1 / (omega + 1)

#         if index_bitstring == "00":
#             expected_coefficient *= int(occupancy_0, 2)
#         elif index_bitstring == "01":
#             expected_coefficient *= int(occupancy_1, 2)
#         elif index_bitstring == "10":
#             expected_coefficient *= np.sqrt(
#                 int(occupancy_1, 2) * (int(occupancy_0, 2) + 1)
#             )
#         elif index_bitstring == "11":
#             expected_coefficient *= np.sqrt(
#                 (int(occupancy_1, 2) + 1) * int(occupancy_0, 2)
#             )

#         assert np.isclose(
#             expected_coefficient * initial_state[int(initial_bitstring, 2)],
#             wavefunction[
#                 int(TOY_BOSONIC_HAMILTONIAN_SELECT_STATE_MAP[initial_bitstring], 2)
#             ],
#         )
#     else:
#         assert np.isclose(
#             initial_state[int(initial_bitstring, 2)],
#             wavefunction[
#                 int(TOY_BOSONIC_HAMILTONIAN_SELECT_STATE_MAP[initial_bitstring], 2)
#             ],
#         )


# @pytest.mark.parametrize("index_state", ["00", "01", "10", "11"])
# @pytest.mark.parametrize("occupancy_1", ["00", "01", "10", "11"])
# @pytest.mark.parametrize("occupancy_0", ["00", "01", "10", "11"])
# def test_select_oracle_on_superposition_state_for_toy_bosonic_hamiltonian(
#     index_state, occupancy_1, occupancy_0
# ):
#     simulator, circuit, intitial_state_of_val_control_index, system = (
#         get_bosonic_select_oracle_test_inputs()
#     )

#     random_fock_state_coeffs = (
#         np.random.uniform(-1, 1, size=1 << system.number_of_system_qubits)
#         + np.random.uniform(-1, 1, size=1 << system.number_of_system_qubits) * 1j
#     )
#     random_fock_state_coeffs /= np.linalg.norm(random_fock_state_coeffs)

#     initial_state_of_system = np.zeros(
#         1 << system.number_of_system_qubits, dtype=np.complex128
#     )
#     for system_state_index in range(1 << system.number_of_system_qubits):
#         initial_state_of_system[system_state_index] = random_fock_state_coeffs[
#             system_state_index
#         ]
#     initial_state = np.kron(
#         intitial_state_of_val_control_index, initial_state_of_system
#     )

#     wavefunction = simulator.simulate(
#         circuit, initial_state=initial_state
#     ).final_state_vector

#     initial_bitstring = (
#         "1" + "00" + "00" + index_state + occupancy_1 + occupancy_0
#     )  # validation, control, index, system
#     assert initial_state[int(initial_bitstring, 2)] != 0

#     if TOY_BOSONIC_HAMILTONIAN_SELECT_STATE_MAP[initial_bitstring][0] == "0":
#         omega = 3
#         expected_coefficient = 1 / (omega + 1)

#         if index_state == "00":
#             expected_coefficient *= int(occupancy_0, 2)
#         elif index_state == "01":
#             expected_coefficient *= int(occupancy_1, 2)
#         elif index_state == "10":
#             expected_coefficient *= np.sqrt(
#                 int(occupancy_1, 2) * (int(occupancy_0, 2) + 1)
#             )
#         elif index_state == "11":
#             expected_coefficient *= np.sqrt(
#                 (int(occupancy_1, 2) + 1) * int(occupancy_0, 2)
#             )

#         assert np.isclose(
#             expected_coefficient * initial_state[int(initial_bitstring, 2)],
#             wavefunction[
#                 int(TOY_BOSONIC_HAMILTONIAN_SELECT_STATE_MAP[initial_bitstring], 2)
#             ],
#         )
#     else:
#         assert np.isclose(
#             initial_state[int(initial_bitstring, 2)],
#             wavefunction[
#                 int(TOY_BOSONIC_HAMILTONIAN_SELECT_STATE_MAP[initial_bitstring], 2)
#             ],
#         )


@pytest.mark.parametrize(
    "operators",
    [
        ParticleOperator("a3^ a2^ a1 a0"),
        (
            ParticleOperator("a0^ a0")
            + ParticleOperator("a1^ a1")
            + ParticleOperator("a0^ a1")
            + ParticleOperator("a1^ a0")
        ),
    ],
)
@pytest.mark.parametrize("maximum_occupation_number", np.random.randint(2, 7, size=3))
@pytest.mark.parametrize("index", np.random.randint(0, 32, 2))
@pytest.mark.parametrize("bosonic_state", np.random.randint(0, 1 << 12, 64))
def test_select_oracle_on_one_two_body_bosonic_terms(
    operators, maximum_occupation_number, index, bosonic_state
):
    number_of_index_qubits = max(int(np.ceil(np.log2(len(operators.to_list())))), 1)
    index = index % len(operators.to_list())
    maximum_mode = operators.max_mode()
    maximum_number_of_bosonic_ops_in_term = max(
        s.count("a") for s in list(operators.op_dict.keys())
    )
    number_of_occupation_qubits = int(np.ceil(np.log2(maximum_occupation_number)))
    number_of_clean_ancillae = (
        maximum_number_of_bosonic_ops_in_term
        + max(number_of_occupation_qubits - 2, 0)
        + 4
    )
    simulator = cirq.Simulator(dtype=complex)
    circuit = cirq.Circuit()

    validation = cirq.LineQubit(0)
    bosonic_rotation_register = [
        cirq.LineQubit(i + 1) for i in range(maximum_number_of_bosonic_ops_in_term)
    ]
    clean_ancillae = [
        cirq.LineQubit(i + 1 + maximum_number_of_bosonic_ops_in_term)
        for i in range(number_of_clean_ancillae)
    ]
    index_register = [
        cirq.LineQubit(
            i + number_of_clean_ancillae + 1 + maximum_number_of_bosonic_ops_in_term
        )
        for i in range(number_of_index_qubits)
    ]
    system = System(
        number_of_modes=maximum_mode + 1,
        maximum_occupation_number=maximum_occupation_number,
        number_of_used_qubits=1
        + number_of_clean_ancillae
        + number_of_index_qubits
        + maximum_number_of_bosonic_ops_in_term,
        has_bosons=True,
    )
    bosonic_state = bosonic_state % (1 << system.number_of_system_qubits)
    circuit.append(cirq.I.on_each(*clean_ancillae))
    circuit.append(cirq.I.on_each(*system.bosonic_system))
    circuit += add_lobe_oracle(
        operators.to_list(),
        validation,
        index_register,
        system,
        bosonic_rotation_register,
        clean_ancillae,
        perform_coefficient_oracle=False,
        decompose=False,
    )

    clean_ancillae_state = np.zeros(
        1 << (number_of_clean_ancillae + maximum_number_of_bosonic_ops_in_term)
    )
    clean_ancillae_state[0] = 1
    initial_state_of_val_and_clean = np.kron(np.array([0, 1]), clean_ancillae_state)

    initial_index_state = np.zeros(1 << number_of_index_qubits)
    initial_index_state[index] = 1
    index_register_bitstring = format(index, f"#0{2+number_of_index_qubits}b")[2:]

    initial_bosonic_state = np.zeros(1 << system.number_of_system_qubits)
    initial_bosonic_state[bosonic_state] = 1
    bosonic_register_bitstring = format(
        bosonic_state, f"#0{2+system.number_of_system_qubits}b"
    )[2:]
    bosonic_registers_bitstrings = [
        bosonic_register_bitstring[
            (i * number_of_occupation_qubits) : (i * number_of_occupation_qubits)
            + number_of_occupation_qubits
        ]
        for i in range(maximum_mode + 1)
    ]
    bosonic_registers_bitstrings = bosonic_registers_bitstrings[::-1]
    expected_bosonic_registers_bitstrings = copy.deepcopy(bosonic_registers_bitstrings)

    initial_state = np.kron(
        np.kron(initial_state_of_val_and_clean, initial_index_state),
        initial_bosonic_state,
    )

    wavefunction = simulator.simulate(
        circuit, initial_state=initial_state
    ).final_state_vector

    term_fired = True
    if (
        len(operators.to_list()[index].split()) == 2
        and operators.to_list()[index].split()[0].mode
        == operators.to_list()[index].split()[1].mode
    ):
        if (
            bosonic_registers_bitstrings[operators.to_list()[index].split()[0].mode]
            == "0" * number_of_occupation_qubits
        ):
            term_fired = False
    else:
        for op in operators.to_list()[index].split():
            if op.creation:
                if (
                    bosonic_registers_bitstrings[op.mode]
                    == "1" * number_of_occupation_qubits
                ):
                    term_fired = False
                else:
                    expected_bosonic_registers_bitstrings[op.mode] = format(
                        int(bosonic_registers_bitstrings[op.mode], 2) + 1,
                        f"#0{2+number_of_occupation_qubits}b",
                    )[2:]
            else:
                if (
                    bosonic_registers_bitstrings[op.mode]
                    == "0" * number_of_occupation_qubits
                ):
                    term_fired = False
                else:
                    expected_bosonic_registers_bitstrings[op.mode] = format(
                        int(bosonic_registers_bitstrings[op.mode], 2) - 1,
                        f"#0{2+number_of_occupation_qubits}b",
                    )[2:]

    expected_bitstring = ""
    if not term_fired:
        expected_bitstring = (
            "1"
            + ("0" * maximum_number_of_bosonic_ops_in_term)
            + ("0" * number_of_clean_ancillae)
            + index_register_bitstring
        )
        for bitstring in bosonic_registers_bitstrings[::-1]:
            expected_bitstring += bitstring
        expected_coefficient = 1
    else:
        expected_bitstring = (
            "0"
            + ("0" * maximum_number_of_bosonic_ops_in_term)
            + ("0" * number_of_clean_ancillae)
            + index_register_bitstring
        )
        for bitstring in expected_bosonic_registers_bitstrings[::-1]:
            expected_bitstring += bitstring

        expected_coefficient = 1 / ((1 << number_of_occupation_qubits)) ** (
            maximum_number_of_bosonic_ops_in_term / 2
        )
        for op in operators.to_list()[index].split():
            if op.creation:
                expected_coefficient *= np.sqrt(
                    int(expected_bosonic_registers_bitstrings[op.mode], 2)
                )
            else:
                expected_coefficient *= np.sqrt(
                    int(bosonic_registers_bitstrings[op.mode], 2)
                )
    assert np.allclose(wavefunction[int(expected_bitstring, 2)], expected_coefficient)
