#!/usr/bin/env python3
"""Updates a neural network using gradient descent with L2 regularization."""

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Updates the weights and biases of a neural network using
    gradient descent with L2 regularization.

    Args:
        Y: One-hot labels of shape (classes, m)
        weights: Dictionary of weights and biases
        cache: Dictionary of activations
        alpha: Learning rate
        lambtha: L2 regularization parameter
        L: Number of layers

    Returns:
        None
    """
    m = Y.shape[1]

    dZ = cache["A{}".format(L)] - Y

    for layer in range(L, 0, -1):

        A_prev = cache["A{}".format(layer - 1)]

        W = weights["W{}".format(layer)].copy()

        dW = (np.matmul(dZ, A_prev.T) / m) + (lambtha / m) * W
        db = np.sum(dZ, axis=1, keepdims=True) / m

        weights["W{}".format(layer)] = W - alpha * dW
        weights["b{}".format(layer)] -= alpha * db

        if layer > 1:
            dA = np.matmul(W.T, dZ)
            A = cache["A{}".format(layer - 1)]
            dZ = dA * (1 - A ** 2)
