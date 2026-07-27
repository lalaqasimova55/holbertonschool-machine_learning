#!/usr/bin/env python3
"""Batch normalization layer creation."""

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """Creates a batch normalization layer in TensorFlow.

    Args:
        prev: activated output of the previous layer
        n: number of nodes in the layer to be created
        activation: activation function to be used on the output

    Returns:
        The activated output for the layer
    """
    initializer = tf.keras.initializers.VarianceScaling(
        mode='fan_avg'
    )

    layer = tf.keras.layers.Dense(
        units=n,
        kernel_initializer=initializer,
        use_bias=False
    )(prev)

    gamma = tf.ones_initializer()
    beta = tf.zeros_initializer()

    batch_norm = tf.keras.layers.BatchNormalization(
        epsilon=1e-7,
        gamma_initializer=gamma,
        beta_initializer=beta
    )(layer)

    if activation is None:
        return batch_norm

    return activation(batch_norm)
