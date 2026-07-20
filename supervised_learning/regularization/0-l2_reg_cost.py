#!/usr/bin/env python3
"""Calculates the cost of a neural network with L2 regularization."""

import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    Calculates the cost of a neural network with L2 regularization.

    Args:
        cost: Cost of the network without regularization.
        lambtha: Regularization parameter.
        weights: Dictionary containing the weights and biases.
        L: Number of layers.
        m: Number of data points.

    Returns:
        Cost with L2 regularization.
    """
    l2_cost = 0

    for i in range(1, L + 1):
        l2_cost += np.sum(np.square(weights["W{}".format(i)]))

    return cost + (lambtha / (2 * m)) * l2_cost
