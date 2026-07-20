#!/usr/bin/env python3
"""Calculates the cost of a Keras model with L2 regularization."""

import tensorflow as tf


def l2_reg_cost(cost, model):
    """
    Calculates the total cost of a model including L2 regularization.

    Args:
        cost: Tensor containing the original cost.
        model: Keras model with L2 regularization.

    Returns:
        Tensor containing the total cost.
    """
    return cost + tf.add_n(model.losses)
