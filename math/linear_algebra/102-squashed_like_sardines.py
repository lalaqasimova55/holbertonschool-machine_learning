#!/usr/bin/env python3
"""Module that concatenates matrices along any axis"""


def cat_matrices(mat1, mat2, axis=0):
    """Concatenates two matrices along a given axis or returns None"""

    # type mismatch
    if type(mat1) != type(mat2):
        return None

    # VECTOR CASE (1D list)
    if isinstance(mat1, list) and mat1 and not isinstance(mat1[0], list):
        if axis == 0:
            return mat1 + mat2
        return None

    # BASE CASE (numbers)
    if not isinstance(mat1, list):
        return mat1 + mat2

    # structure mismatch
    if len(mat1) != len(mat2):
        return None

    # axis 0 → concatenate top level
    if axis == 0:
        return [cat_matrices(m1, m2, 0) for m1, m2 in zip(mat1, mat2)]

    # deeper axis recursion
    result = []
    for i in range(len(mat1)):
        merged = cat_matrices(mat1[i], mat2[i], axis - 1)
        if merged is None:
            return None
        result.append(merged)

    return result
