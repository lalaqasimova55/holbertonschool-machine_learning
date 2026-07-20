#!/usr/bin/env python3
"""Performs batch normalization."""

import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    Normalizes an unactivated output using batch normalization.

    Args:
        Z (numpy.ndarray): Unactivated output of shape (m, n).
        gamma (numpy.ndarray): Scale parameter of shape (1, n).
        beta (numpy.ndarray): Offset parameter of shape (1, n).
        epsilon (float): Small number to avoid division by zero.

    Returns:
        numpy.ndarray: Batch-normalized output.
    """
    mean = np.mean(Z, axis=0, keepdims=True)
    variance = np.var(Z, axis=0, keepdims=True)

    Z_norm = (Z - mean) / np.sqrt(variance + epsilon)
    return gamma * Z_norm + beta
