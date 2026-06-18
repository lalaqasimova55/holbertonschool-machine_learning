#!/usr/bin/env python3
"""Calculates the sensitivity for each class in a confusion matrix"""

import numpy as np


def sensitivity(confusion):
    """
    Calculates the sensitivity for each class

    confusion: numpy.ndarray of shape (classes, classes)
               where rows are actual labels and columns are predictions

    Returns:
        numpy.ndarray of shape (classes,) containing the sensitivity
        of each class
    """
    true_positives = np.diag(confusion)
    actual_positives = np.sum(confusion, axis=1)

    return true_positives / actual_positives
