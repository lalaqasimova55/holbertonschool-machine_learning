#!/usr/bin/env python3
"""Batch normalization"""

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer.
    """
    initializer = tf.keras.initializers.VarianceScaling(
        scale=2.0,
        mode='fan_avg'
    )

    x = tf.keras.layers.Dense(
        units=n,
        kernel_initializer=initializer
    )(prev)

    x = tf.keras.layers.BatchNormalization(
        epsilon=1e-7,
        beta_initializer='zeros',
        gamma_initializer='ones'
    )(x)

    return activation(x)
