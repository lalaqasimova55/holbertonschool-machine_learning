#!/usr/bin/env python3
"""L2 Regularization cost"""

import tensorflow as tf


def l2_reg_cost(cost, model):
    """
    Calculates the cost of a neural network with L2 regularization.

    Args:
        cost: tensor containing the original cost
        model: Keras model with L2 regularization

    Returns:
        Tensor containing the regularization cost for each layer.
    """
    return tf.stack(model.losses)
