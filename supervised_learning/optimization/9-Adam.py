#!/usr/bin/env python3
"""Updates a variable using the Adam optimization algorithm."""

import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon,
                          var, grad, v, s, t):
    """
    Updates a variable using Adam optimization.

    Args:
        alpha (float): Learning rate.
        beta1 (float): Weight for the first moment.
        beta2 (float): Weight for the second moment.
        epsilon (float): Small number to avoid division by zero.
        var (numpy.ndarray): Variable to update.
        grad (numpy.ndarray): Gradient of the variable.
        v (numpy.ndarray): Previous first moment.
        s (numpy.ndarray): Previous second moment.
        t (int): Time step for bias correction.

    Returns:
        tuple: (updated variable, new first moment, new second moment)
    """
    # Update biased first moment estimate
    v = beta1 * v + (1 - beta1) * grad

    # Update biased second raw moment estimate
    s = beta2 * s + (1 - beta2) * (grad ** 2)

    # Compute bias-corrected moments
    v_corr = v / (1 - beta1 ** t)
    s_corr = s / (1 - beta2 ** t)

    # Update variable
    var = var - alpha * v_corr / (np.sqrt(s_corr) + epsilon)

    return var, v, s
