#!/usr/bin/env python3
"""Module that concatenates two numpy arrays along a specific axis"""


import numpy as np


def np_cat(mat1, mat2, axis=0):
    """Returns a new numpy array that is the concatenation of two arrays"""
    return np.concatenate((mat1, mat2), axis=axis)
