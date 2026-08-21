#!/usr/bin/env python3
"""Expectation Maximization Module for GMM"""

import numpy as np

initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5,
                             verbose=False):
    """Performs the expectation maximization algorithm for a GMM.

    Args:
        X: numpy.ndarray of shape (n, d)
        k: positive integer, number of clusters
        iterations: positive integer, maximum number of iterations
        tol: non-negative float, tolerance for early stopping
        verbose: boolean, whether to print log likelihood

    Returns:
        pi, m, S, g, l
        or None, None, None, None, None on failure
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None, None, None

    n, d = X.shape

    if not isinstance(k, int) or k <= 0 or k >= n:
        return None, None, None, None, None

    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None, None

    if not isinstance(tol, (int, float)) or tol < 0:
        return None, None, None, None, None

    if not isinstance(verbose, bool):
        return None, None, None, None, None

    pi, m, S = initialize(X, k)

    if pi is None or m is None or S is None:
        return None, None, None, None, None

    l = None

    for i in range(iterations):
        g, new_l = expectation(X, pi, m, S)

        if g is None or new_l is None:
            return None, None, None, None, None

        if verbose and (i % 10 == 0):
            print("Log Likelihood after {} iterations: {:.5f}".format(
                i, new_l))

        if l is not None and abs(new_l - l) <= tol:
            break

        l = new_l

        pi, m, S = maximization(X, g)

        if pi is None or m is None or S is None:
            return None, None, None, None, None

    return pi, m, S, g, l
