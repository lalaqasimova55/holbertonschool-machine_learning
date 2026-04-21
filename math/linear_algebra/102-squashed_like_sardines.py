#!/usr/bin/env python3
"""Module that concatenates matrices along a specific axis"""


def cat_matrices(mat1, mat2, axis=0):
    """Concatenates two matrices along a given axis or returns None"""

    # both are not same type → fail
    if type(mat1) != type(mat2):
        return None

    # if we reached number level (base case)
    if not isinstance(mat1, list):
        return mat1 if axis == 0 else None

    # axis == 0 → stack lists
    if axis == 0:
        return mat1 + mat2

    # axis > 0 → go deeper recursively
    if len(mat1) != len(mat2):
        return None

    result = []
    for i in range(len(mat1)):
        merged = cat_matrices(mat1[i], mat2[i], axis - 1)
        if merged is None:
            return None
        result.append(merged)

    return result
