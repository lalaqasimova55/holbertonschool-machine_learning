#!/usr/bin/env python3
"""Calculates the specificity for each class in a confusion matrix"""

import numpy as np


def specificity(confusion):
    """
    Calculates the specificity for each class

    confusion: numpy.ndarray of shape (classes, classes)

    Returns:
        numpy.ndarray of shape (classes,) containing the specificity
        of each class
    """
    total = np.sum(confusion)
    classes = confusion.shape[0]

    specificity = np.zeros(classes)

    for i in range(classes):
        tp = confusion[i, i]
        fp = np.sum(confusion[:, i]) - tp
        fn = np.sum(confusion[i, :]) - tp
        tn = total - tp - fp - fn

        specificity[i] = tn / (tn + fp)

    return specificity
