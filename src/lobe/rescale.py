import numpy as np
from copy import deepcopy


def rescale_coefficients(coefficients, rescaling_factors):
    """Rescale the coefficients for a linear combination of block-encodings

    Args:
        coefficients (List[float]): The coefficients of the operators in the linear combination
        rescaling_factors (List[float]): The expected rescaling factors of the block-encodings for each operator

    Returns:
        List[float]: The rescaled coefficients
        float: The overall rescaling factor: sum(|coeff*rescaling_factor|)
    """
    rescaled_coefficients = []
    overall_rescaling_factor = 0

    for coeff, rescaling_factor in zip(coefficients, rescaling_factors):
        rescaled_coefficients.append(coeff * rescaling_factor / max(rescaling_factors))
        overall_rescaling_factor += np.abs(coeff * rescaling_factor)

    return rescaled_coefficients, overall_rescaling_factor


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
