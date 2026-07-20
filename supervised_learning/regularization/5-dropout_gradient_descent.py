#!/usr/bin/env python3
"""Gradient Descent with Dropout"""

import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Updates the weights of a neural network using gradient descent
    with Dropout regularization.

    Parameters:
    Y -- one-hot labels of shape (classes, m)
    weights -- dictionary containing the weights and biases
    cache -- dictionary containing activations and dropout masks
    alpha -- learning rate
    keep_prob -- probability of keeping a neuron active
    L -- number of layers

    Returns:
    None
    """
    m = Y.shape[1]

    dZ = cache["A{}".format(L)] - Y

    for layer in range(L, 0, -1):
        A_prev = cache["A{}".format(layer - 1)]
        W = weights["W{}".format(layer)].copy()

        dW = np.matmul(dZ, A_prev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m

        weights["W{}".format(layer)] = (
            weights["W{}".format(layer)] - alpha * dW
        )
        weights["b{}".format(layer)] = (
            weights["b{}".format(layer)] - alpha * db
        )

        if layer > 1:
            dA = np.matmul(W.T, dZ)

            D = cache["D{}".format(layer - 1)]
            dA = (dA * D) / keep_prob

            A = cache["A{}".format(layer - 1)]
            dZ = dA * (1 - A ** 2)
