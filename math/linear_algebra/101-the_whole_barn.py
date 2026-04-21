#!/usr/bin/env python3
"""Module that adds matrices of any dimension"""


def add_matrices(mat1, mat2):
    """Returns element-wise sum of two matrices or None if mismatch"""

    if isinstance(mat1, list) and isinstance(mat2, list):
        if len(mat1) != len(mat2):
            return None

        result = []
        for i in range(len(mat1)):
            res = add_matrices(mat1[i], mat2[i])
            if res is None:
                return None
            result.append(res)

        return result

    if isinstance(mat1, (int, float)) and isinstance(mat2, (int, float)):
        return mat1 + mat2

    return None
