#!/usr/bin/env python3
"""Maximization step Module for GMM"""


import numpy as np


def maximization(X, g):
    """Calculates the maximization step in the EM algorithm for a GMM.

    Args:
        X: numpy.ndarray of shape (n, d) containing the data set
        g: numpy.ndarray of shape (k, n) containing posterior probabilities

    Returns:
        pi, m, S, or None, None, None on failure
        pi: numpy.ndarray of shape (k,) updated priors
        m: numpy.ndarray of shape (k, d) updated means
        S: numpy.ndarray of shape (k, d, d) updated covariance matrices
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(g, np.ndarray) or len(g.shape) != 2:
        return None, None, None

    n, d = X.shape
    k = g.shape[0]

    if g.shape[1] != n:
        return None, None, None

    # Check if posterior probabilities sum to 1 for each data point
    if not np.isclose(np.sum(g, axis=0), 1.0).all():
        return None, None, None

    # Effective number of points assigned to each cluster
    Nk = np.sum(g, axis=1)

    # Updated priors
    pi = Nk / n

    # Updated means
    m = np.matmul(g, X) / Nk[:, np.newaxis]

    # Updated covariance matrices
    S = np.zeros((k, d, d))
    for i in range(k):
        diff = X - m[i]
        S[i] = np.matmul(g[i] * diff.T, diff) / Nk[i]

    return pi, m, S
