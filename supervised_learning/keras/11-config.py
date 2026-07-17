#!/usr/bin/env python3
"""Save and load a model's configuration"""

import tensorflow.keras as K


def save_config(network, filename):
    """
    Saves a model's configuration in JSON format.

    Args:
        network: the model whose configuration should be saved
        filename: the path to the JSON file

    Returns:
        None
    """
    with open(filename, 'w') as f:
        f.write(network.to_json())


def load_config(filename):
    """
    Loads a model from a JSON configuration file.

    Args:
        filename: the path to the JSON configuration file

    Returns:
        The loaded model
    """
    with open(filename, 'r') as f:
        json_config = f.read()

    return K.models.model_from_json(json_config)
