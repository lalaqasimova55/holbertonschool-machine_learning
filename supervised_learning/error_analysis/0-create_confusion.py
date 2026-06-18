#!/usr/bin/env python3
"""Creates a confusion matrix"""

import numpy as np


def create_confusion_matrix(labels, logits):
    """
    Creates a confusion matrix

    labels: one-hot numpy.ndarray of shape (m, classes)
            containing the correct labels
    logits: one-hot numpy.ndarray of shape (m, classes)
            containing the predicted labels

    Returns:
        confusion matrix of shape (classes, classes)
    """
    classes = labels.shape[1]

    true_labels = np.argmax(labels, axis=1)
    pred_labels = np.argmax(logits, axis=1)

    confusion = np.zeros((classes, classes))

    for true, pred in zip(true_labels, pred_labels):
        confusion[true, pred] += 1

    return confusion
