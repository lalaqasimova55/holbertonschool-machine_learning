#!/usr/bin/env python3
"""Determines the best number of clusters for a GMM using BIC."""

import numpy as np

expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5,
        verbose=False):
    """Calculates the best number of clusters for a GMM using BIC.

    Args:
        X (numpy.ndarray): Data set of shape (n, d).
        kmin (int): Minimum number of clusters.
        kmax (int): Maximum number of clusters.
        iterations (int): Maximum number of EM iterations.
        tol (float): EM convergence tolerance.
        verbose (bool): Whether to print EM information.

    Returns:
        tuple: best_k, best_result, l, b
        or (None, None, None, None) on failure.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None, None

    n, d = X.shape

    if n == 0 or d == 0:
        return None, None, None, None

    if not isinstance(kmin, int) or kmin <= 0:
        return None, None, None, None

    if kmax is None:
        kmax = n

    if not isinstance(kmax, int) or kmax <= 0:
        return None, None, None, None

    if kmin > kmax or kmax > n:
        return None, None, None, None

    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None

    if not isinstance(tol, (int, float)) or tol < 0:
        return None, None, None, None

    ks = range(kmin, kmax + 1)

    l = np.zeros(kmax - kmin + 1)
    b = np.zeros(kmax - kmin + 1)

    results = []

    for i, k in enumerate(ks):
        result = expectation_maximization(
            X,
            k,
            iterations=iterations,
            tol=tol,
            verbose=verbose
        )

        if result is None:
            return None, None, None, None

        pi, m, S, likelihood = result

        if pi is None or m is None or S is None or likelihood is None:
            return None, None, None, None

        # Number of parameters:
        # k means: k * d
        # covariance matrices: k * d * (d + 1) / 2
        # priors: k - 1
        p = k * d + k * d * (d + 1) / 2 + (k - 1)

        l[i] = likelihood
        b[i] = p * np.log(n) - 2 * likelihood

        results.append((pi, m, S))

    best_index = np.argmin(b)
    best_k = kmin + best_index
    best_result = results[best_index]

    return best_k, best_result, l, b
