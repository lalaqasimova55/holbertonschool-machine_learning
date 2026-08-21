#!/usr/bin/env python3

kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    Tests for the optimum number of clusters by variance.
    """

    if not hasattr(X, 'shape') or len(X.shape) != 2:
        return None, None

    n = X.shape[0]

    if not isinstance(kmin, int) or kmin < 1:
        return None, None

    if kmax is None:
        kmax = n - 1

    if not isinstance(kmax, int) or kmax <= kmin or kmax >= n:
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

        if k == kmin:
            first_var = var

        d_vars.append(first_var - var)

    return results, d_vars
