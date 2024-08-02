from copy import deepcopy
import numpy as np
from openparticle import BosonOperator


def rescale_terms(terms, maximum_occupation_number):
    """Rescale the coefficients of the terms to work with LOBE

    The two rescaling constraints that are accounted for are:
        1. All term coefficients must have magnitude <= 1. Therefore, all term
            coefficients are divided by $max(\\alpha_i)$
        2. Bosonic operators pick up a coefficient of $\sqrt{\\frac{n}{\Omega + 1}}$
            (or $n+1$) instead of $\sqrt{n}$ (or $n+1$). Therefore, we block-encode
            a Hamiltonian that is rescaled by a factor of $\\frac{1}{(\Omega+1)^{k/2}}$
            where $k$ is the maximum number of bosonic operators appearing in a single
            term. Terms with $k^\prime < k$ must then be multiplied by a rescaling
            factor of $\\frac{1}{(\Omega + 1)^{k - k^\prime}}$ in order to abide by
            this rescaling.

    NOTE: The terms returned by this function will NOT give a rescaled version of the
        original Hamiltonian due to constraint #2.

    Args:
        terms (List[ParticleOperator]): The terms comprising the original Hamiltonian
            given as a linear combination of ladder operators.
        maximum_occupation_number (int): The maximum number of particles that can
            occupy a single bosonic mode ($\Omega$).

    Returns:
        List[ParticleOperator]: The rescaled terms to be used for LOBE
        float: The rescaling factor s.t. $\frac{H}{scaling_factor} = \bar{H}$
    """
    max_coeff = max([np.abs(term.coeff) for term in terms])

    numbers_of_bosonic_ops = get_numbers_of_bosonic_operators_in_terms(terms)

    rescaled_terms = []
    for i, term in enumerate(deepcopy(terms)):
        scale = max_coeff
        scale *= (maximum_occupation_number + 1) ** (
            (max(numbers_of_bosonic_ops) - numbers_of_bosonic_ops[i]) / 2
        )
        term = (1 / scale) * term
        rescaled_terms.append(term)

    scaling_factor = max_coeff * (
        (maximum_occupation_number + 1) ** ((max(numbers_of_bosonic_ops)) / 2)
    )

    return rescaled_terms, scaling_factor


def get_numbers_of_bosonic_operators_in_terms(terms):
    """Get a list of the number of bosonic operators in each term.

    Args:
        terms (List[ParticleOperator]): The terms comprising the original Hamiltonian
            given as a linear combination of ladder operators.

    Returns:
        List[int]: A list of the number of bosonic operators in each term
    """
    numbers_of_bosonic_ops = []
    for term in terms:
        numbers_of_bosonic_ops.append(0)
        for operator in term.parse():
            if isinstance(operator, BosonOperator):
                numbers_of_bosonic_ops[-1] += 1

    return numbers_of_bosonic_ops
