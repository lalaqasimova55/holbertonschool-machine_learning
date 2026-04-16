#!/usr/bin/env python3
"""
Module that computes derivative of a polynomial.
"""


def poly_derivative(poly):
    """
    Returns derivative of a polynomial represented as a list.

    poly: list of coefficients
    """

    if (not isinstance(poly, list) or len(poly) == 0):
        return None

    for c in poly:
        if not isinstance(c, (int, float)):
            return None

    result = []
    for i in range(1, len(poly)):
        result.append(poly[i] * i)

    if len(result) == 0:
        return [0]

    if all(x == 0 for x in result):
        return [0]

    return result
