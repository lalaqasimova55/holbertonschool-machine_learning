#!/usr/bin/env python3
"""Calculates the cost of a neural network with L2 regularization."""

import tensorflow as tf


def l2_reg_cost(cost, model):
    """
    Calculates the cost of a neural network with L2 regularization.

    Args:
        cost: Tensor containing the cost without L2 regularization.
        model: Keras model with L2 regularization.

    Returns:
        Tensor containing the total cost for each layer.
    """
    return cost + tf.stack(model.losses)
