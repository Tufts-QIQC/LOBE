from copy import deepcopy
import numpy as np
from openparticle import BosonOperator, OccupationOperator


def bosonically_rescale_terms(terms, maximum_occupation_number):
    """Rescale the coefficients of the terms for bosonic operators

    Bosonic operators pick up a coefficient of $\sqrt{\\frac{n}{\Omega + 1}}$
        (or $n+1$) instead of $\sqrt{n}$ (or $n+1$). Therefore, we block-encode
        a Hamiltonian that is rescaled by a factor of $\\frac{1}{(\Omega+1)^{k/2}}$
        where $k$ is the maximum number of bosonic operators appearing in a single
        term. Terms with $k^\prime < k$ must then be multiplied by a rescaling
        factor of $\\frac{1}{(\Omega + 1)^{k - k^\prime}}$ in order to abide by
        this rescaling.

    NOTE: The terms returned by this function will NOT give a rescaled version of the
        original Hamiltonian

    Args:
        terms (List[ParticleOperator]): The terms comprising the original Hamiltonian
            given as a linear combination of ladder operators.
        maximum_occupation_number (int): The maximum number of particles that can
            occupy a single bosonic mode ($\Omega$).

    Returns:
        List[ParticleOperator]: The rescaled terms to be used for LOBE
        float: The rescaling factor s.t. $\frac{H}{scaling_factor} = \bar{H}$
    """
    numbers_of_bosonic_ops = get_numbers_of_bosonic_operators_in_terms(terms)

    rescaled_terms = []
    for i, term in enumerate(deepcopy(terms)):
        scale = (maximum_occupation_number + 1) ** (
            (max(numbers_of_bosonic_ops) - numbers_of_bosonic_ops[i]) / 2
        )
        term = (1 / scale) * term
        rescaled_terms.append(term)

    scaling_factor = (maximum_occupation_number + 1) ** (
        (max(numbers_of_bosonic_ops)) / 2
    )

    return rescaled_terms, scaling_factor


def rescale_terms_usp(terms):
    """Rescale the coefficients of the terms such that all have magnitude less than 1

    Args:
        terms (List[ParticleOperator]): The terms comprising the original Hamiltonian
            given as a linear combination of ladder operators.

    Returns:
        List[ParticleOperator]: The rescaled terms to be used for LOBE
        float: The rescaling factor s.t. $\frac{H}{scaling_factor} = \bar{H}$
    """
    max_coeff = max([np.abs(term.coeff) for term in terms])
    rescaled_terms = [(1 / max_coeff) * term for term in deepcopy(terms)]
    return rescaled_terms, max_coeff


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
            elif (
                isinstance(operator, OccupationOperator)
                and operator.particle_type == "a"
            ):
                numbers_of_bosonic_ops[-1] += 2 * operator.power

    return numbers_of_bosonic_ops


def get_active_bosonic_modes(operator):
    """Get a list of the bosonic modes being acted on.

    Args:
        operator (Optional[ParticleOperator, List[ParticleOperator]]): The operator/term in question

    Returns:
        List[int]: A list of active bosonic modes
    """
    active_modes = []
    for term in operator.to_list():
        for op in term.split():
            if isinstance(op, BosonOperator):
                if op.mode not in active_modes:
                    active_modes.append(op.mode)
    return active_modes


def get_number_of_active_bosonic_modes(terms):
    """Get a list of the number of bosonic modes being acted on in each term.

    Args:
        terms (List[ParticleOperator]): The terms comprising the original Hamiltonian
            given as a linear combination of ladder operators.

    Returns:
        List[int]: A list of the number of bosonic operators in each term
    """
    numbers_of_active_bosonic_modes = []
    for term in terms:
        active_modes = get_active_bosonic_modes(term)
        numbers_of_active_bosonic_modes.append(len(active_modes))

    return numbers_of_active_bosonic_modes
