#!/usr/bin/env python3
"""Sets up the RMSProp optimization algorithm in TensorFlow."""

import tensorflow as tf


def create_RMSProp_op(alpha, beta2, epsilon):
    """
    Creates the TensorFlow RMSProp optimizer.

    Args:
        alpha (float): Learning rate.
        beta2 (float): RMSProp weight (discounting factor).
        epsilon (float): Small number to avoid division by zero.

    Returns:
        tf.keras.optimizers.RMSprop: RMSProp optimizer.
    """
    return tf.keras.optimizers.RMSprop(
        learning_rate=alpha,
        rho=beta2,
        epsilon=epsilon
    )
