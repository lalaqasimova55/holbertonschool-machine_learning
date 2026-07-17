#!/usr/bin/env python3
"""Make predictions using a neural network"""

import tensorflow.keras as K


def predict(network, data, verbose=False):
    """
    Makes a prediction using a neural network.

    Args:
        network: the network model to use
        data: the input data
        verbose: whether to print progress during prediction

    Returns:
        The predictions for the input data
    """
    return network.predict(data, verbose=verbose)
