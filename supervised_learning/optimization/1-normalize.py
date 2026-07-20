#!/usr/bin/env python3
"""Normalizes a matrix."""

import numpy as np


def normalize(X, m, s):
    """
    Normalizes a matrix.

    Args:
        X (numpy.ndarray): Matrix of shape (d, nx).
        m (numpy.ndarray): Mean of each feature.
        s (numpy.ndarray): Standard deviation of each feature.

    Returns:
        numpy.ndarray: The normalized matrix.
    """
    return (X - m) / s
