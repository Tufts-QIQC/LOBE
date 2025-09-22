import cirq

def add_ancilla_reflection(ancilla_register, ctrls=([], [])):
    '''
    Implements R = 2|0><0| - 1
    Inputs:
        Ancilla register the rotation acts about 
    Returns: 
        List[gates] that implement R on the ancilla_register
    '''

    gates = []

    if len(ancilla_register) == 1:
        return [cirq.Z.on(ancilla_register[0]).controlled_by(*ctrls[0], control_values=ctrls[1])]
    

    gates.append(
        cirq.X.on_each(*ancilla_register)
    )

    gates.append(cirq.H.on(ancilla_register[0]).controlled_by(*ctrls[0], control_values=ctrls[1]))
    controls = ancilla_register[1:]
    controls += ctrls[0]
    values = [1]*len(ancilla_register[1:])
    values += ctrls[1]
    gates.append(cirq.X.on(ancilla_register[0]).controlled_by(*controls, control_values=values),)
    gates.append(cirq.H.on(ancilla_register[0]).controlled_by(*ctrls[0], control_values=ctrls[1]))

    gates.append(
        cirq.Moment(cirq.X.on_each(*ancilla_register))
    )

    return gates