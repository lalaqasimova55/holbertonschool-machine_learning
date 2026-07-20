#!/usr/bin/env python3
"""Sets up the momentum optimization algorithm in TensorFlow."""

import tensorflow as tf


def create_momentum_op(alpha, beta1):
    """
    Creates the TensorFlow momentum optimizer.

    Args:
        alpha (float): Learning rate.
        beta1 (float): Momentum weight.

    Returns:
        tf.keras.optimizers.SGD: Momentum optimizer.
    """
    return tf.keras.optimizers.SGD(
        learning_rate=alpha,
        momentum=beta1
    )
