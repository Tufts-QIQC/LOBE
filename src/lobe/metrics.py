class CircuitMetrics:

    def __init__(self):
        self.number_of_elbows = 0
        self.number_of_rotations = 0
        self.number_of_t_gates = 0
        self.clean_ancillae_usage = [0]

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
        for number_of_used_ancillae in other.clean_ancillae_usage:
            joined_metrics.clean_ancillae_usage.append(
                self.clean_ancillae_usage[-1] + number_of_used_ancillae
            )

        return joined_metrics

    def add_to_clean_ancillae_usage(self, change):
        self.clean_ancillae_usage.append(self.clean_ancillae_usage[-1] + change)
