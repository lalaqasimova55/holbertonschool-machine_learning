#!/usr/bin/env python3
"""
Builds a neural network with the Keras Functional API
"""

import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Builds a neural network with the Keras library.

    Args:
        nx: number of input features
        layers: list containing the number of nodes in each layer
        activations: list containing the activation functions
        lambtha: L2 regularization parameter
        keep_prob: probability that a node will be kept

    Returns:
        A Keras Model
    """
    inputs = K.Input(shape=(nx,))
    x = inputs

    for i in range(len(layers)):
        x = K.layers.Dense(
            units=layers[i],
            activation=activations[i],
            kernel_regularizer=K.regularizers.l2(lambtha)
        )(x)

        if i != len(layers) - 1:
            x = K.layers.Dropout(rate=1 - keep_prob)(x)

    model = K.Model(inputs=inputs, outputs=x)

    return model
