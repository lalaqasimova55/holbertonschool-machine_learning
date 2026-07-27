#!/usr/bin/env python3
"""Batch Normalization"""

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer for a neural network.

    Args:
        prev: Activated output of the previous layer
        n: Number of nodes in the layer
        activation: Activation function

    Returns:
        Activated output tensor
    """
    init = tf.keras.initializers.VarianceScaling(mode='fan_avg')

    x = tf.keras.layers.Dense(
        units=n,
        kernel_initializer=init
    )(prev)

    x = tf.keras.layers.BatchNormalization(
        epsilon=1e-7,
        beta_initializer='zeros',
        gamma_initializer='ones'
    )(x)

    return activation(x)
