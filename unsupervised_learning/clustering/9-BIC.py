#!/usr/bin/env python3
"""Bayesian Information Criterion (BIC) Module for GMM"""


import numpy as np

expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """Finds the best number of clusters for a GMM using BIC.

    Args:
        X: numpy.ndarray of shape (n, d)
        kmin: positive integer, minimum clusters to check
        kmax: positive integer, maximum clusters to check
        iterations: positive integer for EM
        tol: non-negative float tolerance for EM
        verbose: boolean for EM log print

    Returns:
        best_k, best_result, l, b or None, None, None, None on failure
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None

    n, d = X.shape

    if not isinstance(kmin, int) or kmin <= 0 or kmin > n:
        return None, None, None, None

    if kmax is None:
        kmax = n

    if not isinstance(kmax, int) or kmax <= 0 or kmax > n:
        return None, None, None, None

    if kmin > kmax:
        return None, None, None, None

    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None

    if not isinstance(tol, (int, float)) or tol < 0:
        return None, None, None, None

    if not isinstance(verbose, bool):
        return None, None, None, None

    l_arr = np.zeros(kmax - kmin + 1)
    b_arr = np.zeros(kmax - kmin + 1)
    results = []

    for idx, k in enumerate(range(kmin, kmax + 1)):
        pi, m, S, g, log_l = expectation_maximization(
            X, k, iterations, tol, verbose
        )
        if pi is None:
            return None, None, None, None

        results.append((pi, m, S))
        l_arr[idx] = log_l

        # Calculate number of free parameters p
        p = (k - 1) + (k * d) + (k * d * (d + 1) / 2)

        # Calculate BIC
        b_arr[idx] = p * np.log(n) - 2 * log_l

    best_idx = np.argmin(b_arr)
    best_k = kmin + best_idx
    best_result = results[best_idx]

    return best_k, best_result, l_arr, b_arr
