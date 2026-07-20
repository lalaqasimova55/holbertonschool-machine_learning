#!/usr/bin/env python3
"""Creates a batch normalization layer."""

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer for a neural network.

    Args:
        prev: Activated output of the previous layer.
        n (int): Number of nodes in the new layer.
        activation: Activation function.

    Returns:
        Tensor containing the activated output.
    """
    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')

    dense = tf.keras.layers.Dense(
        units=n,
        kernel_initializer=initializer,
        use_bias=False
    )

    x = dense(prev)

    x = tf.keras.layers.BatchNormalization(
        axis=-1,
        momentum=0.99,
        epsilon=1e-7,
        beta_initializer='zeros',
        gamma_initializer='ones'
    )(x)

    return activation(x)
