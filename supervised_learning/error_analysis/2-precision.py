#!/usr/bin/env python3
"""Calculates the precision for each class in a confusion matrix"""

import numpy as np


def precision(confusion):
    """
    Calculates the precision for each class

    confusion: numpy.ndarray of shape (classes, classes)
               where rows are actual labels and columns are predictions

    Returns:
        numpy.ndarray of shape (classes,) containing the precision
        of each class
    """
    true_positives = np.diag(confusion)
    predicted_positives = np.sum(confusion, axis=0)

    return true_positives / predicted_positives
