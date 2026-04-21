#!/usr/bin/env python3
"""Module that slices a numpy.ndarray along specific axes"""


import numpy as np


def np_slice(matrix, axes={}):
    """Returns a sliced version of a numpy array based on axes dictionary"""
    slices = [slice(None)] * matrix.ndim

    for axis, s in axes.items():
        slices[axis] = slice(*s)

    return matrix[tuple(slices)]
