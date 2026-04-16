#!/usr/bin/env python3
"""
Module that computes sum of squares from 1 to n.
"""


def summation_i_squared(n):
    """
    Returns the sum of i^2 from 1 to n.
    If n is not valid, returns None.
    """
    if not isinstance(n, int) or n <= 0:
        return None

    return (n * (n + 1) * (2 * n + 1)) // 6
