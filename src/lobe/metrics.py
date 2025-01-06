class CircuitMetrics:

    def __init__(self):
        self.number_of_elbows = 0
        self.number_of_rotations = 0
        self.number_of_t_gates = 0
        self.clean_ancillae_usage = []

    def __add__(self, other):

        joined_metrics = CircuitMetrics()
        joined_metrics.number_of_elbows += (
            self.number_of_elbows + other.number_of_elbows
        )
        joined_metrics.number_of_rotations += (
            self.number_of_rotations + other.number_of_rotations
        )
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
