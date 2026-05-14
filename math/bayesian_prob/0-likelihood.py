#!/usr/bin/env python3
"""Likelihood function for a binomial distribution"""

import numpy as np
from math import factorial


def likelihood(x, n, P):
    """
    Calculates the likelihood of obtaining this data
    given various hypothetical probabilities of success

    Parameters:
    x -- number of patients with severe side effects
    n -- total number of patients
    P -- 1D numpy.ndarray of hypothetical probabilities

    Returns:
    numpy.ndarray containing the likelihood for each probability in P
    """

    # Validate n
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    # Validate x
    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0"
        )

    # Validate x <= n
    if x > n:
        raise ValueError("x cannot be greater than n")

    # Validate P
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    # Validate values in P
    if np.any(P < 0) or np.any(P > 1):
        raise ValueError("All values in P must be in the range [0, 1]")

    # Binomial coefficient
    coeff = factorial(n) / (factorial(x) * factorial(n - x))

    # Likelihood calculation
    L = coeff * (P ** x) * ((1 - P) ** (n - x))

    return L
