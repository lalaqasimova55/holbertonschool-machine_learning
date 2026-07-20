#!/usr/bin/env python3
"""Calculates the normalization constants for a matrix."""

import numpy as np


def normalization_constants(X):
    """
    Calculates the normalization constants of a matrix.

    Args:
        X (numpy.ndarray): Matrix of shape (m, nx) containing the data.

    Returns:
        tuple: (mean, std)
            mean is a numpy.ndarray of shape (nx,) containing the mean
            of each feature.
            std is a numpy.ndarray of shape (nx,) containing the standard
            deviation of each feature.
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    return mean, std
