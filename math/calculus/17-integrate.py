#!/usr/bin/env python3
"""Polynomial integration module"""


def poly_integral(poly, C=0):
    """Compute the integral of a polynomial"""
    if not isinstance(poly, list) or len(poly) == 0:
        return None
    if not isinstance(C, (int, float)):
        return None
    if not all(isinstance(c, (int, float)) for c in poly):
        return None

    res = [C]

    for i, c in enumerate(poly):
        res.append(c / (i + 1))

    while len(res) > 1 and res[-1] == 0:
        res.pop()

    for i in range(len(res)):
        if isinstance(res[i], float) and res[i].is_integer():
            res[i] = int(res[i])

    return res
