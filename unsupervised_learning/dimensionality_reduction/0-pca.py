#!/usr/bin/env python3
"""PCA"""


import numpy as np


def pca(X, var=0.95):
    """Perform PCA on a dataset.

    Args:
        X: numpy.ndarray of shape (n, d)
        var: fraction of variance to maintain

    Returns:
        W: weights matrix of shape (d, nd)
    """
    U, S, Vt = np.linalg.svd(X, full_matrices=False)

    variance = S ** 2
    cumulative_variance = np.cumsum(variance)
    total_variance = np.sum(variance)

    nd = np.searchsorted(
        cumulative_variance / total_variance,
        var
    ) + 1

    return Vt.T[:, :nd]
