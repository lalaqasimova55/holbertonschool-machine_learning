#!/usr/bin/env python3
"""Save and load a Keras model"""

import tensorflow.keras as K


def save_model(network, filename):
    """
    Saves an entire model

    Args:
        network: the model to save
        filename: path to save the model

    Returns:
        None
    """
    network.save(filename)


def load_model(filename):
    """
    Loads an entire model

    Args:
        filename: path to the saved model

    Returns:
        The loaded model
    """
    return K.models.load_model(filename)
