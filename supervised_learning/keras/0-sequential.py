#!/usr/bin/env python3
"""
Builds a neural network with the Keras library
"""

import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Builds a neural network using Keras.

    Args:
        nx: number of input features
        layers: list containing the number of nodes in each layer
        activations: list containing the activation functions for each layer
        lambtha: L2 regularization parameter
        keep_prob: probability that a node will be kept

    Returns:
        A Keras Sequential model
    """
    model = K.models.Sequential()

    for i in range(len(layers)):
        if i == 0:
            model.add(
                K.layers.Dense(
                    layers[i],
                    activation=activations[i],
                    kernel_regularizer=K.regularizers.l2(lambtha),
                    input_shape=(nx,)
                )
            )
        else:
            model.add(
                K.layers.Dense(
                    layers[i],
                    activation=activations[i],
                    kernel_regularizer=K.regularizers.l2(lambtha)
                )
            )

        if i != len(layers) - 1:
            model.add(K.layers.Dropout(rate=1 - keep_prob))

    return model
