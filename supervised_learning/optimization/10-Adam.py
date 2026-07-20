#!/usr/bin/env python3
"""Sets up the Adam optimization algorithm in TensorFlow."""

import tensorflow as tf


def create_Adam_op(alpha, beta1, beta2, epsilon):
    """
    Creates the TensorFlow Adam optimizer.

    Args:
        alpha (float): Learning rate.
        beta1 (float): Weight for the first moment.
        beta2 (float): Weight for the second moment.
        epsilon (float): Small number to avoid division by zero.

    Returns:
        tf.keras.optimizers.Adam: Adam optimizer.
    """
    return tf.keras.optimizers.Adam(
        learning_rate=alpha,
        beta_1=beta1,
        beta_2=beta2,
        epsilon=epsilon
    )
