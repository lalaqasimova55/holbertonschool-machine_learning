#!/usr/bin/env python3
"""Updates a variable using the RMSProp optimization algorithm."""

import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    Updates a variable using RMSProp.

    Args:
        alpha (float): Learning rate.
        beta2 (float): RMSProp weight.
        epsilon (float): Small number to avoid division by zero.
        var (numpy.ndarray): Variable to update.
        grad (numpy.ndarray): Gradient of the variable.
        s (numpy.ndarray): Previous second moment.

    Returns:
        tuple: (updated variable, new second moment)
    """
    s = beta2 * s + (1 - beta2) * (grad ** 2)
    var = var - alpha * grad / (np.sqrt(s) + epsilon)

    return var, s
