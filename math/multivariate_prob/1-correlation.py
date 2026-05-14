#!/usr/bin/env python3
"""Calculates correlation matrix from covariance matrix"""

import numpy as np


def correlation(C):
    """
    Calculates a correlation matrix from a covariance matrix

    Args:
        C: numpy.ndarray of shape (d, d) covariance matrix

    Returns:
        numpy.ndarray of shape (d, d) correlation matrix
    """

    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")

    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")

    d = C.shape[0]

    # standard deviations (sqrt of diagonal)
    std = np.sqrt(np.diag(C))

    # outer product of stds
    denom = np.outer(std, std)

    return C / denom
