#!/usr/bin/env python3
"""
Creates a batch normalization layer for a neural network
"""

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer.

    Args:
        prev: Activated output of the previous layer
        n: Number of nodes in the layer
        activation: Activation function

    Returns:
        The activated output tensor
    """
    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')

    dense = tf.keras.layers.Dense(
        units=n,
        kernel_initializer=initializer
    )(prev)

    batch_norm = tf.keras.layers.BatchNormalization(
        axis=-1,
        momentum=0.99,
        epsilon=1e-7,
        beta_initializer=tf.keras.initializers.Zeros(),
        gamma_initializer=tf.keras.initializers.Ones()
    )(dense)

    return activation(batch_norm)
