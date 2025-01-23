import numpy as np

CLIFFORD_ROTATION_ANGLES = [i * np.pi / 2 for i in range(9)]


class CircuitMetrics:

    def __init__(self):
        self.number_of_elbows = 0
        self.number_of_t_gates = 0
        self.clean_ancillae_usage = []
        self.rotation_angles = []

    def __add__(self, other):

        joined_metrics = CircuitMetrics()
        joined_metrics.number_of_elbows += (
            self.number_of_elbows + other.number_of_elbows
        )
        joined_metrics.rotation_angles += self.rotation_angles + other.rotation_angles
        joined_metrics.number_of_t_gates += (
            self.number_of_t_gates + other.number_of_t_gates
        )
        joined_metrics.clean_ancillae_usage += self.clean_ancillae_usage
        for i, number_of_used_ancillae in enumerate(other.clean_ancillae_usage):
            if i == 0:
                previous = 0
            else:
                previous = other.clean_ancillae_usage[i - 1]
            joined_metrics.add_to_clean_ancillae_usage(
                number_of_used_ancillae - previous
            )

        return joined_metrics

    def add_to_clean_ancillae_usage(self, change):
        if len(self.clean_ancillae_usage) == 0:
            self.clean_ancillae_usage.append(change)
        else:
            self.clean_ancillae_usage.append(self.clean_ancillae_usage[-1] + change)

    def ancillae_highwater(self):
        if len(self.clean_ancillae_usage) != 0:
            return max(self.clean_ancillae_usage)
        else:
            return 0

    @property
    def number_of_nonclifford_rotations(self):
        number_of_nonclifford_rotations = 0
        for angle in self.rotation_angles:
            if not np.any(
                [
                    np.isclose((angle) % (4 * np.pi), clifford_angle)
                    for clifford_angle in CLIFFORD_ROTATION_ANGLES
                ]
            ):
                # Count only nonClifford rotations
                number_of_nonclifford_rotations += 1
        return number_of_nonclifford_rotations

    def display_metrics(self):
        print("--- Metrics ---")
        print("Number of elbows: ", self.number_of_elbows)
        print("Number of T-gates: ", self.number_of_t_gates)
        print(
            "Number of non-Clifford rotations: ", self.number_of_nonclifford_rotations
        )
        print("---------------")
