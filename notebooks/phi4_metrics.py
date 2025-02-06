from openparticle.utils import get_fock_basis, generate_matrix
from openparticle.hamiltonians.phi4_hamiltonian import phi4_Hamiltonian
import cirq
import matplotlib.pyplot as plt
import numpy as np
from functools import partial
import sys, os

sys.path.append(os.path.join(os.path.dirname(os.path.realpath("__file__")), ".."))

from src.lobe.system import System
from src.lobe.lcu import LCU
from src.lobe.asp import get_target_state, add_prepare_circuit
from src.lobe.rescale import (
    get_number_of_active_bosonic_modes,
)
from src.lobe._utils import get_bosonic_exponents
from src.lobe.index import index_over_terms
from src.lobe.metrics import CircuitMetrics
from src.lobe.bosonic import (
    bosonic_product_plus_hc_block_encoding,
    bosonic_product_block_encoding,
)
from tests._utils import _validate_block_encoding

from colors import *


def phi4_lcu_circuit_metrics(resolution, maximum_occupation_number):
    print("---", resolution, "---", maximum_occupation_number, "---")
    operator = phi4_Hamiltonian(resolution, g=1, mb=1).normal_order()
    operator.remove_identity()
    lcu = LCU(
        operator, max_bosonic_occupancy=maximum_occupation_number, zero_threshold=1e-9
    )
    ctrls = ([cirq.LineQubit(-1000000)], [1])
    circuit = cirq.Circuit()
    circuit += cirq.X.on(ctrls[0][0])
    circuit += lcu.get_circuit(ctrls=ctrls)
    circuit += cirq.X.on(ctrls[0][0])

    system = System(
        operator.max_bosonic_mode + 1,
        maximum_occupation_number,
        1000,
        False,
        False,
        True,
    )
    # _validate_block_encoding(
    #     circuit,
    #     system,
    #     lcu.one_norm,
    #     operator,
    #     len(lcu.index_register),
    #     maximum_occupation_number,
    #     using_pytest=False,
    # )

    return lcu.circuit_metrics, lcu.one_norm, len(lcu.index_register), circuit


def phi4_LOBE_circuit_metrics(resolution, maximum_occupation_number):
    operator = phi4_Hamiltonian(resolution, 1, 1).normal_order()
    operator.remove_identity()
    print("---", resolution, "---", maximum_occupation_number, "---")

    grouped_terms = operator.group()

    number_of_block_encoding_ancillae = max(
        get_number_of_active_bosonic_modes(grouped_terms)
    )
    if len(grouped_terms) != len(operator.to_list()):
        number_of_block_encoding_ancillae += 1  # For grouped terms, need an additional BE ancilla to index between the group

    # print(resolution, number_of_block_encoding_ancillae)
    index_register = [
        cirq.LineQubit(-i - 2) for i in range(int(np.ceil(np.log2(len(grouped_terms)))))
    ]
    block_encoding_ancillae = [
        cirq.LineQubit(-100 - i - len(index_register))
        for i in range(number_of_block_encoding_ancillae)
    ]

    ctrls = ([cirq.LineQubit(0)], [1])
    clean_ancillae = [cirq.LineQubit(i + 100) for i in range(100)]
    system = System(
        operator.max_bosonic_mode + 1,
        maximum_occupation_number,
        1000,
        False,
        False,
        True,
    )

    block_encoding_functions = []
    rescaling_factors = []
    for term in grouped_terms:
        plus_hc = False
        if len(term) == 2:
            plus_hc = True
            term = term.to_list()[0]
        active_modes, exponents = get_bosonic_exponents(
            term, operator.max_bosonic_mode + 1
        )

        if not plus_hc:
            block_encoding_functions.append(
                partial(
                    bosonic_product_block_encoding,
                    system=system,
                    block_encoding_ancillae=block_encoding_ancillae,
                    active_indices=active_modes,
                    exponents_list=exponents,
                    clean_ancillae=clean_ancillae[1:],
                )
            )
            rescaling_factors.append(
                np.sqrt(maximum_occupation_number) ** (sum(sum(np.asarray(exponents))))
            )
        else:
            block_encoding_functions.append(
                partial(
                    bosonic_product_plus_hc_block_encoding,
                    system=system,
                    block_encoding_ancillae=block_encoding_ancillae,
                    active_indices=active_modes,
                    exponents_list=exponents,
                    clean_ancillae=clean_ancillae[1:],
                )
            )
            rescaling_factors.append(
                2
                * np.sqrt(maximum_occupation_number)
                ** (sum(sum(np.asarray(exponents))))
            )

    rescaled_coefficients = []
    for term, rescaling_factor in zip(grouped_terms, rescaling_factors):
        rescaled_coefficients.append(
            term.coeffs[0] * rescaling_factor / max(rescaling_factors)
        )

    target_state = get_target_state(rescaled_coefficients)
    gates = []
    metrics = CircuitMetrics()
    for mode in system.bosonic_system:
        for qubit in mode:
            gates.append(cirq.I.on(qubit))

    gates.append(cirq.X.on(ctrls[0][0]))

    _gates, _metrics = add_prepare_circuit(
        index_register, target_state, clean_ancillae=clean_ancillae
    )
    print("Metrics from add_prepare_circuit: \n")
    _metrics.display_metrics()
    gates += _gates
    metrics += _metrics

    _gates, _metrics = index_over_terms(
        index_register, block_encoding_functions, clean_ancillae, ctrls=ctrls
    )
    print("Metrics from index_over_terms: \n")
    _metrics.display_metrics()  # Print metrics
    gates += _gates
    metrics += _metrics

    _gates, _metrics = add_prepare_circuit(
        index_register, target_state, dagger=True, clean_ancillae=clean_ancillae
    )
    print("Metrics from add_prepare_circuit: \n")
    _metrics.display_metrics()
    gates += _gates
    metrics += _metrics

    gates.append(cirq.X.on(ctrls[0][0]))

    overall_rescaling_factor = sum(
        [
            term.coeffs[0] * rescaling_factor
            for term, rescaling_factor in zip(grouped_terms, rescaling_factors)
        ]
    )
    print("Total metrics of the block encoding: \n")
    metrics.display_metrics()
    _validate_block_encoding(
        cirq.Circuit(gates),
        system,
        overall_rescaling_factor,
        operator,
        len(index_register) + number_of_block_encoding_ancillae,
        maximum_occupation_number,
        using_pytest=False,
    )

    return (
        metrics,
        overall_rescaling_factor,
        len(index_register) + number_of_block_encoding_ancillae,
        cirq.Circuit(gates),
    )


def _get_phi4_hamiltonian_norm(res, maximum_occupation_number, g=1):
    ham = phi4_Hamiltonian(res, g=g, mb=1)
    basis = get_fock_basis(ham, maximum_occupation_number)
    matrix = generate_matrix(ham, basis)

    # vals = np.linalg.eigvalsh(matrix)
    # return max(np.abs(vals))
    return np.linalg.norm(matrix, ord=2)


def plot_phi4_changing_occupancy():
    omegas = [int(2**n - 1) for n in np.arange(1, 5, 1)]
    resolution = 2
    print("LOBE")

    LOBE_DATA = [phi4_LOBE_circuit_metrics(resolution, omega) for omega in omegas]
    print("LCU")
    LCU_DATA = [phi4_lcu_circuit_metrics(resolution, omega) for omega in omegas]

    operator_norms = []
    for omega in omegas:
        operator = phi4_Hamiltonian(resolution, 1, 1).normal_order()
        operator.remove_identity()
        operator_norms.append(_get_phi4_hamiltonian_norm(resolution, omega))
        print(resolution, omega, operator_norms[-1])
    system_qubits = [
        System(
            phi4_Hamiltonian(omega, 1, 1).max_bosonic_mode,
            omega,
            1000,
            False,
            False,
            True,
        ).number_of_system_qubits
        for omega in omegas
    ]

    fig, axes = plt.subplots(3, 2, figsize=(16 / 2.54, 18 / 2.54))

    axes[0][0].plot(
        omegas,
        [4 * LOBE_DATA[i][0].number_of_elbows for i in range(len(omegas))],
        color=BLUE,
        marker="o",
        alpha=0.5,
        label="LOBE",
    )
    axes[0][0].plot(
        omegas,
        [4 * LCU_DATA[i][0].number_of_elbows for i in range(len(omegas))],
        color=ORANGE,
        marker="^",
        alpha=0.5,
        label="LCU",
    )
    axes[0][0].set_ylabel("T-gates")
    axes[0][0].set_xlabel("Occupancy ($\Omega$)")

    axes[0][1].plot(
        omegas,
        [LOBE_DATA[i][0].number_of_nonclifford_rotations for i in range(len(omegas))],
        color=BLUE,
        marker="o",
        alpha=0.5,
        label="LOBE",
    )
    axes[0][1].plot(
        omegas,
        [LCU_DATA[i][0].number_of_nonclifford_rotations for i in range(len(omegas))],
        color=ORANGE,
        marker="^",
        alpha=0.5,
        label="LCU",
    )
    axes[0][1].set_ylabel("Rotations")
    axes[0][1].set_xlabel("Occupancy ($\Omega$)")

    axes[1][0].plot(
        omegas,
        [LOBE_DATA[i][2] for i in range(len(omegas))],
        color=BLUE,
        marker="o",
        alpha=0.5,
        label="LOBE",
    )
    axes[1][0].plot(
        omegas,
        [LCU_DATA[i][2] for i in range(len(omegas))],
        color=ORANGE,
        marker="^",
        alpha=0.5,
        label="LCU",
    )
    axes[1][0].set_ylabel("BE-Ancillae")
    axes[1][0].set_xlabel("Occupancy ($\Omega$)")

    axes[1][1].plot(
        omegas,
        [LOBE_DATA[i][1] for i in range(len(omegas))],
        color=BLUE,
        marker="o",
        alpha=0.5,
        label="LOBE",
    )
    axes[1][1].plot(
        omegas,
        [LCU_DATA[i][1] for i in range(len(omegas))],
        color=ORANGE,
        marker="^",
        alpha=0.5,
        label="LCU",
    )
    axes[1][1].plot(
        omegas,
        [operator_norms[i] for i in range(len(omegas))],
        color="black",
        marker="x",
        ls="--",
        alpha=0.5,
    )
    axes[1][1].set_ylabel("Rescaling Factor")
    axes[1][1].set_xlabel("Occupancy ($\Omega$)")

    axes[2][0].plot(
        omegas,
        [
            LCU_DATA[i][0].ancillae_highwater() + LCU_DATA[i][2] + system_qubits[i] + 1
            for i in range(len(omegas))
        ],
        color=ORANGE,
        marker="^",
        alpha=0.5,
        label="LCU",
    )
    axes[2][0].plot(
        omegas,
        [
            LOBE_DATA[i][0].ancillae_highwater()
            + LOBE_DATA[i][2]
            + system_qubits[i]
            + 1
            for i in range(len(omegas))
        ],
        color=BLUE,
        marker="o",
        alpha=0.5,
        label="LOBE",
    )
    axes[2][0].plot(
        [], [], color="black", marker="x", ls="--", alpha=0.5, label="Hamiltonian Norm"
    )
    axes[2][0].set_ylabel("Total Qubit Highwater")
    axes[2][0].set_xlabel("Occupancy ($\Omega$)")

    fig.delaxes(axes[2][1])
    plt.tight_layout()
    axes[2][0].legend(
        loc="upper center",
        bbox_to_anchor=(1.5, 0.75),
        fancybox=True,
        shadow=True,
        ncol=1,
    )
    plt.show()


def plot_phi4_changing_resolution():
    omega = 3
    resolutions = np.arange(2, 8, 1)
    size_of_basis = [
        len(get_fock_basis(phi4_Hamiltonian(res, 1, 1))) for res in resolutions
    ]
    print(size_of_basis)
    print("LOBE")

    LOBE_DATA = [phi4_LOBE_circuit_metrics(res, omega) for res in resolutions]
    print("LCU")
    LCU_DATA = [phi4_lcu_circuit_metrics(res, omega) for res in resolutions]

    operator_norms = []
    for res in resolutions:
        operator = phi4_Hamiltonian(res, 1, 1).normal_order()
        operator.remove_identity()
        operator_norms.append(_get_phi4_hamiltonian_norm(res, omega))
        print(res, omega, operator_norms[-1])
    system_qubits = [
        System(
            phi4_Hamiltonian(res, 1, 1).max_bosonic_mode,
            omega,
            1000,
            False,
            False,
            True,
        ).number_of_system_qubits
        for res in resolutions
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16 / 2.54, 8 / 2.54))

    axes[0][0].plot(
        resolutions,
        [4 * LOBE_DATA[i][0].number_of_elbows for i in range(len(resolutions))],
        color=BLUE,
        marker="o",
        alpha=0.5,
        label="LOBE",
    )
    axes[0][0].plot(
        resolutions,
        [4 * LCU_DATA[i][0].number_of_elbows for i in range(len(resolutions))],
        color=ORANGE,
        marker="^",
        alpha=0.5,
        label="LCU",
    )
    axes[0][0].set_ylabel("T-gates")
    axes[0][0].set_xticklabels([])

    axes[1][0].plot(
        resolutions,
        [
            LOBE_DATA[i][0].number_of_nonclifford_rotations
            for i in range(len(resolutions))
        ],
        color=BLUE,
        marker="o",
        alpha=0.5,
        label="LOBE",
    )
    axes[1][0].plot(
        resolutions,
        [
            LCU_DATA[i][0].number_of_nonclifford_rotations
            for i in range(len(resolutions))
        ],
        color=ORANGE,
        marker="^",
        alpha=0.5,
        label="LCU",
    )
    axes[1][0].set_ylabel("Non-Clifford Rotations")
    axes[1][0].set_xlabel("Resolution ($K$)")

    axes[0][1].plot(
        resolutions,
        [LOBE_DATA[i][2] for i in range(len(resolutions))],
        color=BLUE,
        marker="o",
        alpha=0.5,
        label="LOBE",
    )
    axes[0][1].plot(
        resolutions,
        [LCU_DATA[i][2] for i in range(len(resolutions))],
        color=ORANGE,
        marker="^",
        alpha=0.5,
        label="LCU",
    )
    axes[0][1].set_ylabel("BE Ancillae")
    axes[0][1].set_xticklabels([])
    axes[0][1].set_yticks([0, 10, 20])

    axes[1][1].plot(
        resolutions,
        [
            LCU_DATA[i][0].ancillae_highwater() + LCU_DATA[i][2] + system_qubits[i] + 1
            for i in range(len(resolutions))
        ],
        color=ORANGE,
        marker="^",
        alpha=0.5,
        label="LCU",
    )
    axes[1][1].plot(
        resolutions,
        [
            LOBE_DATA[i][0].ancillae_highwater()
            + LOBE_DATA[i][2]
            + system_qubits[i]
            + 1
            for i in range(len(resolutions))
        ],
        color=BLUE,
        marker="o",
        alpha=0.5,
        label="LOBE",
    )
    axes[1][1].set_ylabel("Max Qubit Usage")
    axes[1][1].set_xlabel("Resolution ($K$)")
    axes[1][1].set_yticks([0, 20, 40])

    axes[1][2].plot(
        resolutions,
        [LOBE_DATA[i][1] for i in range(len(resolutions))],
        color=BLUE,
        marker="o",
        alpha=0.5,
        label="LOBE",
    )
    axes[1][2].plot(
        resolutions,
        [LCU_DATA[i][1] for i in range(len(resolutions))],
        color=ORANGE,
        marker="^",
        alpha=0.5,
        label="LCU",
    )
    axes[1][2].plot(
        resolutions,
        [operator_norms[i] for i in range(len(resolutions))],
        color="black",
        marker="x",
        ls="--",
        alpha=0.5,
    )
    axes[1][2].set_ylabel("Rescaling Factor")
    axes[1][2].set_xlabel("Resolution ($K$)")

    fig.delaxes(axes[0][2])
    plt.tight_layout()
    # axes[2][0].legend(
    #     loc="upper center",
    #     bbox_to_anchor=(1.5, 0.75),
    #     fancybox=True,
    #     shadow=True,
    #     ncol=1,
    # )
    plt.savefig("../manuscript/figures/phi4-resolution.pdf", dpi=300)


# plot_phi4_changing_occupancy()
plot_phi4_changing_resolution()
