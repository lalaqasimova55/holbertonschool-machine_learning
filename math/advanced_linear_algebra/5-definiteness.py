#!/usr/bin/env python3
"""
Module to calculate the definiteness of a matrix
"""
import numpy as np


def definiteness(matrix):
    """
    Calculates the definiteness of a matrix
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1]:
        return None

    if matrix.size == 0:
        return None

    if not np.allclose(matrix, matrix.T):
        return None

    try:
        eigenvalues = np.linalg.eigvals(matrix)

        pos = np.all(eigenvalues > 1e-10)
        neg = np.all(eigenvalues < -1e-10)
        pos_semi = np.all(eigenvalues >= -1e-10) and not pos
        neg_semi = np.all(eigenvalues <= 1e-10) and not neg
        has_pos = np.any(eigenvalues > 1e-10)
        has_neg = np.any(eigenvalues < -1e-10)

        if pos:
            return "Positive definite"
        if neg:
            return "Negative definite"
        if pos_semi:
            return "Positive semi-definite"
        if neg_semi:
            return "Negative semi-definite"
        if has_pos and has_neg:
            return "Indefinite"

        return None
    except Exception:
        return None
