#!/usr/bin/env python3
"""Determines the best number of clusters for a GMM using BIC."""

import numpy as np

expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5,
        verbose=False):
    """Finds the best number of clusters for a GMM using BIC."""

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

    l = np.zeros(kmax - kmin + 1)
    b = np.zeros(kmax - kmin + 1)

    results = []

    for i, k in enumerate(range(kmin, kmax + 1)):
        result = expectation_maximization(
            X,
            k,
            iterations=iterations,
            tol=tol,
            verbose=verbose
        )

        if result is None:
            return None, None, None, None

        pi, m, S, g, likelihood = result

        p = (
            k * d
            + k * d * (d + 1) / 2
            + k - 1
        )

        l[i] = likelihood
        b[i] = p * np.log(n) - 2 * likelihood

        results.append((pi, m, S))

    best_index = np.argmin(b)
    best_k = kmin + best_index
    best_result = results[best_index]

    return best_k, best_result, l, b
