import numpy as np

CLIFFORD_ROTATION_ANGLES = np.array([i * np.pi / 2 for i in range(9)])


class CircuitMetrics:
    """Object containing relevant circuit metrics to count."""

    def __init__(self):
        self.number_of_elbows = 0
        self.number_of_t_gates = 0
        self.clean_ancillae_usage = []
        self.rotation_angles = []

    def __add__(self, other):
        self.number_of_elbows += other.number_of_elbows
        self.rotation_angles += other.rotation_angles
        self.number_of_t_gates += other.number_of_t_gates
        previous = 0
        for number_of_used_ancillae in other.clean_ancillae_usage:
            self.add_to_clean_ancillae_usage(number_of_used_ancillae - previous)
            previous = number_of_used_ancillae
        return self

    def add_to_clean_ancillae_usage(self, change):
        """Account for clean ancillae being used or freed

        Args:
            - change (int): The number of clean ancillae used (positive) or freed (negative)
        """
        if len(self.clean_ancillae_usage) == 0:
            self.clean_ancillae_usage.append(change)
        else:
            self.clean_ancillae_usage.append(self.clean_ancillae_usage[-1] + change)

    def ancillae_highwater(self):
        """Get the maximum number of clean ancillae used at any point in time during the circuit

        Returns:
            - int: The maximum number of clean ancillae
        """
        if len(self.clean_ancillae_usage) != 0:
            return max(self.clean_ancillae_usage)
        else:
            return 0

    @property
    def number_of_nonclifford_rotations(self, slow_count=True, decimals=12):
        """The number of arbitrary rotations that are not merely Clifford operations"""
        return (
            len(self.rotation_angles)
            - np.isin(
                (np.array(self.rotation_angles) % (4 * np.pi)).round(decimals),
                CLIFFORD_ROTATION_ANGLES.round(decimals),
            ).sum()
        )

    def display_metrics(self):
        """Print relevant metrics to screen"""
        print("--- Metrics ---")
        print("Number of elbows: ", self.number_of_elbows)
        print("Number of T-gates: ", self.number_of_t_gates)
        print(
            "Number of non-Clifford rotations: ", self.number_of_nonclifford_rotations
        )
        print("Maximum number of used clean ancillae: ", self.ancillae_highwater())
        print("---------------")
