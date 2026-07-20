#!/usr/bin/env python3
"""Create a layer with Dropout."""

import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob, training=True):
    """
    Creates a layer of a neural network using dropout.

    Args:
        prev: output of previous layer
        n: number of nodes
        activation: activation function
        keep_prob: probability of keeping a neuron
        training: whether the model is training

    Returns:
        Output tensor
    """

    initializer = tf.keras.initializers.VarianceScaling(
        scale=2.0,
        mode="fan_avg"
    )

    dense = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=initializer
    )

    x = dense(prev)

    x = tf.keras.layers.Dropout(
        rate=1 - keep_prob
    )(x, training=training)

    return x
