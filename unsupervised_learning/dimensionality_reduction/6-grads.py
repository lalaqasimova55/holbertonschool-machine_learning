#!/usr/bin/env python3
"""Calculate gradients for t-SNE."""

import numpy as np

Q_affinities = __import__('5-Q_affinities').Q_affinities


def grads(Y, P):
    """Calculate the gradients of Y.

    Args:
        Y: numpy.ndarray of shape (n, ndim), low-dimensional data.
        P: numpy.ndarray of shape (n, n), P affinities.

    Returns:
        dY: numpy.ndarray of shape (n, ndim), gradients.
        Q: numpy.ndarray of shape (n, n), Q affinities.
    """
    Q, _ = Q_affinities(Y)

    n = Y.shape[0]

    # Pairwise differences: Y_i - Y_j
    diff = Y[:, np.newaxis, :] - Y[np.newaxis, :, :]

    # Student-t kernel numerator
    Y_squared = np.sum(Y ** 2, axis=1, keepdims=True)
    D = Y_squared + Y_squared.T - 2 * np.matmul(Y, Y.T)
    num = 1 / (1 + D)

    # P_ij - Q_ij
    PQ = P - Q

    # Gradient
    dY = np.sum(
        2 * PQ[:, :, np.newaxis] * num[:, :, np.newaxis] * diff,
        axis=1
    )

    return dY, Q
