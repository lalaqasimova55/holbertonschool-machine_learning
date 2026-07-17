#!/usr/bin/env python3
"""
Sets up Adam optimization for a Keras model
"""

import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """
    Configures a Keras model for training.

    Args:
        network: the Keras model to optimize
        alpha: learning rate
        beta1: first Adam optimization parameter
        beta2: second Adam optimization parameter

    Returns:
        None
    """
    optimizer = K.optimizers.Adam(
        learning_rate=alpha,
        beta_1=beta1,
        beta_2=beta2
    )

    network.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
