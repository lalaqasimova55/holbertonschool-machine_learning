#!/usr/bin/env python3
"""Module that adds two matrices of any dimension"""


def add_matrices(mat1, mat2):
    """Returns a new matrix resulting from element-wise addition"""

    if type(mat1) != type(mat2):
        return None

    if isinstance(mat1, list):
        if len(mat1) != len(mat2):
            return None
        return [add_matrices(m1, m2) for m1, m2 in zip(mat1, mat2)]

    return mat1 + mat2
