#!/usr/bin/env python3
"""Shuffles two matrices in the same way."""

import numpy as np


def shuffle_data(X, Y):
    """
    Shuffles the data points in two matrices the same way.

    Args:
        X (numpy.ndarray): First matrix of shape (m, nx).
        Y (numpy.ndarray): Second matrix of shape (m, ny).

    Returns:
        tuple: (X_shuffled, Y_shuffled)
    """
    permutation = np.random.permutation(X.shape[0])

    return X[permutation], Y[permutation]
