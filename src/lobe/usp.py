import cirq


def add_naive_usp(index_register):
    return cirq.Moment(cirq.H.on_each(*index_register))
