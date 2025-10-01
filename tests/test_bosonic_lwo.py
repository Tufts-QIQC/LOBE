import cirq
import pytest
import numpy as np
from functools import partial
from openparticle import ParticleOperator, generate_matrix

import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath('__file__')), '../'))
from src.lobe.self_inverse_boson import self_inverse_bosonic_number_operator_block_encoding, self_inverse_bosonic_product_plus_hc_block_encoding
from src.lobe.system import System
from _utils import get_basis_of_full_system


@pytest.mark.parametrize("maximum_occupation_number", [1, 3, 7])
def test_self_inverse_bosonic_number_operator_block_encoding_on_single_mode_satisfies_walker_conditions(maximum_occupation_number):

    hamiltonian_operator = (
        ParticleOperator('a0^ a0')
    )
    rescaling_factor = maximum_occupation_number
    basis = get_basis_of_full_system(maximum_occupation_number + 1, 0, hamiltonian_operator.max_bosonic_mode + 1)
    H_matrix = generate_matrix(hamiltonian_operator, basis)

    index_register = [cirq.LineQubit(0)]
    clean_ancilla_register = [cirq.LineQubit(i) for i in range(1, 7 + 1)]
    be_anc_register = [cirq.LineQubit(8), cirq.LineQubit(9)]
    system_register = System(maximum_occupation_number, 
                        len(index_register) + len(clean_ancilla_register) + len(be_anc_register),
                        0, 
                        1)
    
    gates = []

    PREPARE = [cirq.X.on(index_register[0])]
    gates += PREPARE

    SELECT = self_inverse_bosonic_number_operator_block_encoding(
        system = system_register,
        block_encoding_ancillae=be_anc_register,
        active_mode = 0,
        sign = 1,
        clean_ancillae=clean_ancilla_register,
        ctrls = ([index_register[0]], [1])
    )

    gates += SELECT

    gates += [cirq.X.on(index_register[0])] #PREP^DAGGER


    assert np.allclose(
        H_matrix,
        (cirq.Circuit(
            PREPARE,
            SELECT,
            PREPARE
        ).unitary()*rescaling_factor)[:maximum_occupation_number + 1,
             :maximum_occupation_number + 1]
    ) #<0|PREP^\DAGGER SEL PREP|0> = H/ALPHA
    
    assert np.allclose(
        np.eye(maximum_occupation_number + 1),
        (cirq.Circuit(
            PREPARE,
            SELECT,
            SELECT,
            PREPARE
        ).unitary())[:maximum_occupation_number + 1,
             :maximum_occupation_number + 1]
    )#<0|PREP^\DAGGER SEL^2 PREP|0> = 1