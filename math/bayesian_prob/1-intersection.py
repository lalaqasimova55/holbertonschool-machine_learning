#!/usr/bin/env python3
"""Calculates the intersection of obtaining the data"""

import numpy as np


def intersection(x, n, P, Pr):
    """
    Calculates the intersection of obtaining this data
    with the various hypothetical probabilities

    Args:
        x: number of patients that develop severe side effects
        n: total number of patients observed
        P: 1D numpy.ndarray containing hypothetical probabilities
        Pr: 1D numpy.ndarray containing prior beliefs of P

    Returns:
        A 1D numpy.ndarray containing the intersection
        of obtaining x and n with each probability in P
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

    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError(
            "Pr must be a numpy.ndarray with the same shape as P"
        )

    if np.any(P < 0) or np.any(P > 1):
        raise ValueError(
            "All values in P must be in the range [0, 1]"
        )

    if np.any(Pr < 0) or np.any(Pr > 1):
        raise ValueError(
            "All values in Pr must be in the range [0, 1]"
        )

    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    fact_n = 1
    fact_x = 1
    fact_n_x = 1

    for i in range(1, n + 1):
        fact_n *= i

    for i in range(1, x + 1):
        fact_x *= i

    for i in range(1, (n - x) + 1):
        fact_n_x *= i

    coeff = fact_n / (fact_x * fact_n_x)

    likelihood = coeff * (P ** x) * ((1 - P) ** (n - x))

    return likelihood * Pr
