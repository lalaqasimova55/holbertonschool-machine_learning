#!/usr/bin/env python3
"""
Determines the optimum number of clusters for K-means.
"""

import numpy as np

kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    Determines the optimum number of clusters.

    Args:
        X (numpy.ndarray): Dataset of shape (n, d).
        kmin (int): Minimum number of clusters.
        kmax (int): Maximum number of clusters.
        iterations (int): Maximum number of K-means iterations.

    Returns:
        tuple: (results, d_vars), or (None, None) on failure.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None

    n = X.shape[0]

    if n < 1 or kmin <= 0:
        return None, None

    if kmax is None:
        kmax = kmin + 1

    if not isinstance(kmax, int) or kmax <= kmin or kmax > n:
        return None, None

    if not isinstance(kmin, int) or kmin > n:
        return None, None

    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    results = []
    d_vars = []

    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations)

        if C is None or clss is None:
            return None, None

        results.append((C, clss))

        var = variance(X, C)

        if var is None:
            return None, None

        if k == kmin:
            base_var = var

        d_vars.append(base_var - var)

    return results, d_vars
