#!/usr/bin/env python3
"""Calculates the continuous posterior probability"""

from scipy import special


def posterior(x, n, p1, p2):
    """
    Calculates the posterior probability that p is within
    the range [p1, p2] given x and n

    Args:
        x: number of patients with severe side effects
        n: total number of patients observed
        p1: lower bound of the range
        p2: upper bound of the range

    Returns:
        The posterior probability that p is within [p1, p2]
    """

    if type(n) is not int or n <= 0:
        raise ValueError("n must be a positive integer")

    if type(x) is not int or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0"
        )

    if x > n:
        raise ValueError("x cannot be greater than n")

    if type(p1) is not float or p1 < 0 or p1 > 1:
        raise ValueError("p1 must be a float in the range [0, 1]")

    if type(p2) is not float or p2 < 0 or p2 > 1:
        raise ValueError("p2 must be a float in the range [0, 1]")

    if p2 <= p1:
        raise ValueError("p2 must be greater than p1")

    a = x + 1
    b = n - x + 1

    return special.betainc(a, b, p2) - special.betainc(a, b, p1)
