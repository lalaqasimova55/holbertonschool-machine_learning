#!/usr/bin/env python3
"""Batch normalization"""

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer.
    
    Args:
        prev: Tensor representing the activated output of the previous layer
        n: Integer, number of nodes in the layer to be created
        activation: Activation function that should be used on the output
        
    Returns:
        The activated output for the layer
    """
    initializer = tf.keras.initializers.VarianceScaling(
        scale=2.0,
        mode='fan_avg'
    )

    # Dense layer before BN (use_bias=False if BN provides offset)
    x = tf.keras.layers.Dense(
        units=n,
        kernel_initializer=initializer,
        use_bias=False
    )(prev)

    # Batch Normalization prior to activation
    x = tf.keras.layers.BatchNormalization(
        epsilon=1e-7
    )(x)

    # Apply activation function LAST
    if activation is not None:
        x = activation(x)

    return x
