#!/usr/bin/env python3
"""Save and load model weights"""

import tensorflow.keras as K


def save_weights(network, filename, save_format='keras'):
    """
    Saves a model's weights

    Args:
        network: the model whose weights should be saved
        filename: the path where the weights will be saved
        save_format: the format to save the weights

    Returns:
        None
    """
    network.save_weights(filename)


def load_weights(network, filename):
    """
    Loads a model's weights

    Args:
        network: the model into which the weights will be loaded
        filename: the path to the saved weights

    Returns:
        None
    """
    network.load_weights(filename)
