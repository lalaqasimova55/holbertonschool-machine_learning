#!/usr/bin/env python3
"""Calculates the posterior probability"""

import numpy as np


def posterior(x, n, P, Pr):
    """
    Calculates the posterior probability for the various
    hypothetical probabilities given the data

    Args:
        x: number of patients with severe side effects
        n: total number of patients observed
        P: 1D numpy.ndarray of hypothetical probabilities
        Pr: 1D numpy.ndarray of prior beliefs for P

    Returns:
        A 1D numpy.ndarray containing the posterior probabilities
    """

    if type(n) is not int or n <= 0:
        raise ValueError("n must be a positive integer")

    if type(x) is not int or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0"
        )

    if x > n:
        raise ValueError("x cannot be greater than n")

    if type(P) is not np.ndarray or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    if type(Pr) is not np.ndarray or Pr.shape != P.shape:
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

    coeff = 1

    for i in range(1, x + 1):
        coeff *= (n - i + 1) / i

    likelihood = coeff * (P ** x) * ((1 - P) ** (n - x))

    intersection = likelihood * Pr

    marginal = np.sum(intersection)

    return intersection / marginal
