import cirq


def diffusion_operator(index_register):
    """Add a series of Hadamards implementing the Diffusion operator

    Args:
        - index_register (List[cirq.LineQubit]): The register on which to apply the Hadamards

    Returns:
        - cirq.Moment: The gates applying the Hadamards onto the index register"""
    return cirq.Moment(cirq.H.on_each(*index_register))
