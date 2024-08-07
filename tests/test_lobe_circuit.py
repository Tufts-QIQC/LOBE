from openparticle import *
from src.lobe.lobe_circuit import lobe_circuit
import pytest


def test_lobe_function_toy_fermionic():
    hamiltonian = ParticleOperator({"b0^ b0": 1, "b0^ b1": 1, "b1^ b0": 1, "b1^ b1": 1})

    _, unitary, matrix = lobe_circuit(hamiltonian)
    assert np.allclose(np.real(unitary), matrix)


@pytest.mark.parametrize("max_bose_occ", [1, 3])
def test_lobe_function_toy_bosonic(max_bose_occ):
    hamiltonian = ParticleOperator({"a0^ a0": 1, "a0^ a1": 1, "a1^ a0": 1, "a1^ a1": 1})

    _, unitary, matrix = lobe_circuit(hamiltonian, max_bose_occ=max_bose_occ)
    assert np.allclose(np.real(unitary), matrix)


def test_lobe_function_quartic_osc():
    renormalized_quartic_oscillator_hamiltonian = ParticleOperator(
        {
            "a0^ a0^ a0^ a0": 2.849565686667622,
            "a0^ a0^": 3.5923247590513974,
            "a0^ a0^ a0 a0": -5.536050711865201,
            "a0^ a0": 11.737092068070014,
            "a0^ a0 a0 a0": 2.849565686667622,
            # ' ': 2.4780255664185606, # remove identity term
            "a0 a0": 3.5923247590513974,
            "a0^ a0^ a0^ a0 a0 a0": 3.9998612759007717,
            "a0^ a0^ a0 a0 a0 a0": 3.5581095542809806,
        }
    )

    _, unitary, matrix = lobe_circuit(
        renormalized_quartic_oscillator_hamiltonian, max_bose_occ=3
    )
    assert np.allclose(np.real(unitary), matrix)
