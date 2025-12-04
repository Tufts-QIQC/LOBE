from src.lobe._utils import (
    get_bosonic_exponents,
    translate_antifermions_to_fermions,
)
import numpy as np
from src.lobe._utils import get_bosonic_exponents, translate_antifermions_to_fermions
from src.lobe.metrics import CircuitMetrics

CLIFFORD_ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi]


def remove_clifford_rotations(angles_list, tol: float = 1e-3):
    non_clifford_angles = []

    for angle in angles_list:
        angle_is_not_clifford = True  # assume at first the angle is clifford
        for clifford_angle in CLIFFORD_ANGLES:
            if np.allclose(
                angle, clifford_angle, rtol=tol
            ):  # For each clifford angle, check if the rotation angle is close to clifford within tolerance
                angle_is_not_clifford = False
        if angle_is_not_clifford:
            non_clifford_angles.append(angle)

    return np.array(non_clifford_angles)


def get_rotation_angles(exponents, max_occupancy):
    rotation_angles = []
    for R_and_S in exponents:
        Ri, Si = R_and_S[0], R_and_S[1]
        argument = 1
        for omega in range(Si, max_occupancy - Ri + 1):
            for r in range(0, Ri):
                argument *= np.sqrt(omega - r) / np.sqrt(max_occupancy)
            for s in range(1, Si + 1):
                argument *= np.sqrt(omega - Ri + s) / np.sqrt(max_occupancy)
            angle = 2 * np.arccos(argument)
            rotation_angles.append(angle)

    return rotation_angles


def get_unique_modes(operator):
    """
    Given a ParticleOperator, this function returns a list of unique fermionic modes and a list of unique bosonic modes
    """

    fermionic_modes = []
    bosonic_modes = []

    for term in operator.op_dict.keys():
        for op in term:
            mode = op[1]
            if op[0] == 0:
                fermionic_modes.append(mode)
            elif op[0] == 2:
                bosonic_modes.append(mode)
            else:
                raise Exception(
                    "This function assumes all antifermionic operators are mapped to fermionic ones"
                )

    return list(set(fermionic_modes)), list(set(bosonic_modes))


def separate_bosonic_ops(term):
    bosonic_ops = 1
    assert len(term.to_list()) == 1  # Only pass in a term

    for op in term.split():
        if op.has_bosons:
            bosonic_ops *= op
    return bosonic_ops


def count_metrics(operator, max_occupancy: int = 1):

    groups = operator.group()

    if operator.has_antifermions:
        translated_groups = []
        max_fermionic_mode = operator.max_fermionic_mode
        for term in groups:
            translated_groups.append(
                translate_antifermions_to_fermions(term, max_fermionic_mode)
            )
        groups = translated_groups

    metrics = CircuitMetrics()

    B = 0

    for term in groups:
        if len(term) == 1:
            if term.has_fermions:
                metrics.rescaling_factor += 1
                metrics.number_of_t_gates += 4
                metrics.number_of_elbows += 1
                metrics.clean_ancillae_usage += [1]
                metrics.rotation_angles += []
            elif term.has_bosons:
                metrics.rescaling_factor += max_occupancy
                metrics.number_of_elbows += np.ceil(np.log2(max_occupancy))
                metrics.number_of_t_gates += 7 * np.ceil(np.log2(max_occupancy + 1))
                metrics.clean_ancillae_usage += [
                    i for i in range(1, int(np.ceil(np.log2(max_occupancy))) + 1)
                ]
                metrics.rotation_angles += [
                    2 * np.arccos(np.sqrt(omega * (omega - 1)) / max_occupancy)
                    for omega in range(0, max_occupancy)
                ]
        elif len(term) > 1:
            first_term = term.to_list()[0]

            # assume form of two fermionic ops and 1 or 2 bosonic ops plus h.c.
            term_B_fermion, term_B_boson = get_unique_modes(
                first_term
            )  # B = number of unique bosonic modes
            n_boson_ops = first_term.n_bosons

            # Determine rotations
            exponents = []
            if first_term.has_bosons:
                bosonic_first_term = separate_bosonic_ops(first_term)
                _, exponents = get_bosonic_exponents(
                    bosonic_first_term, bosonic_first_term.max_mode + 1
                )
                metrics.rotation_angles += get_rotation_angles(exponents, max_occupancy)

            if first_term.has_fermions:
                rescaling_factor = max_occupancy ** (n_boson_ops / 2)
                if n_boson_ops == 1:
                    clean_ancillae_usage = [
                        i
                        for i in range(
                            1, int(np.ceil(np.log2(max_occupancy + 1))) + 1 + 1
                        )
                    ]
                    n_t_gates = 12 * np.ceil(np.log2(max_occupancy))
                elif n_boson_ops > 1:
                    clean_ancillae_usage = [
                        i
                        for i in range(1, int(np.ceil(np.log2(max_occupancy))) + 1 + 1)
                    ]
                    n_t_gates = 24 * np.ceil(np.log2(max_occupancy)) - 8
                elif n_boson_ops == 0:
                    clean_ancillae_usage = [i for i in range(1, term_B_fermion - 1 + 1)]
                    n_t_gates = 4 * (term_B_fermion - 1)

            else:
                # assume form of n bosonic ops + h.c.
                term_W = np.ceil(np.log2(max_occupancy + 1))
                rescaling_factor = 2 * (max_occupancy ** (n_boson_ops / 2))
                clean_ancillae_usage = [
                    i for i in range(1, int(np.ceil(np.log2(max_occupancy))) + 1 + 1)
                ]
                n_t_gates = 12 * term_B_boson * term_W - 8 * term_B_boson + 4

            B = max(max(term_B_boson), B)
            metrics.number_of_t_gates += n_t_gates
            metrics.clean_ancillae_usage += clean_ancillae_usage
            metrics.rescaling_factor += rescaling_factor

    L = len(groups)
    metrics.number_of_t_gates += 4 * (L - 1)

    metrics.number_of_be_ancillae = np.ceil(np.log2(L)) + 1 * operator.has_fermions + B

    return metrics
