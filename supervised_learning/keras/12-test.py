#!/usr/bin/env python3
"""Test a neural network"""

import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """
    Tests a neural network.

    Args:
        network: the model to test
        data: input data
        labels: correct one-hot labels
        verbose: whether to print progress during testing

    Returns:
        A list containing the loss and accuracy of the model
    """
    return network.evaluate(data, labels, verbose=verbose)
