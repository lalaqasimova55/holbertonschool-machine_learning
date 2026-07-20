#!/usr/bin/env python3
"""Create a neural network layer using Dropout."""

import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob, training=True):
    """
    Creates a layer of a neural network using dropout.

    Args:
        prev: Tensor containing the output of the previous layer.
        n: Number of nodes in the new layer.
        activation: Activation function.
        keep_prob: Probability that a node will be kept.
        training: Boolean indicating whether the model is in training mode.

    Returns:
        The output tensor of the new layer.
    """
    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')

    dense = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=initializer
    )

    x = dense(prev)

    dropout = tf.keras.layers.Dropout(rate=1 - keep_prob)

    return dropout(x, training=training)
