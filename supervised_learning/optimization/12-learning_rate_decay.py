#!/usr/bin/env python3
"""Creates a learning rate decay operation in TensorFlow."""

import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, decay_step):
    """
    Creates a learning rate schedule using inverse time decay.

    Args:
        alpha (float): Original learning rate.
        decay_rate (float): Learning rate decay factor.
        decay_step (int): Number of steps before each decay.

    Returns:
        tf.keras.optimizers.schedules.InverseTimeDecay:
            Learning rate schedule.
    """
    return tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=alpha,
        decay_steps=decay_step,
        decay_rate=decay_rate,
        staircase=True
    )
