#!/usr/bin/env python3
"""Likelihood function for a binomial distribution"""

import numpy as np


def likelihood(x, n, P):
    """
    Calculates the likelihood of obtaining the data
    given various hypothetical probabilities

    Args:
        x: number of patients with severe side effects
        n: total number of patients observed
        P: 1D numpy.ndarray containing hypothetical probabilities

    Returns:
        A 1D numpy.ndarray containing the likelihood of
        obtaining the data for each probability in P
    """

    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0"
        )

    if x > n:
        raise ValueError("x cannot be greater than n")

    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    if np.any(P < 0) or np.any(P > 1):
        raise ValueError(
            "All values in P must be in the range [0, 1]"
        )

    # Compute n choose x manually
    numerator = 1
    denominator = 1

    for i in range(1, x + 1):
        numerator *= (n - i + 1)
        denominator *= i

    coeff = numerator / denominator

    return coeff * (P ** x) * ((1 - P) ** (n - x))
