#!/usr/bin/env python3
"""Forward propagation with Dropout"""

import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Conducts forward propagation using Dropout.

    Parameters:
        X: numpy.ndarray of shape (nx, m)
        weights: dictionary containing the weights and biases
        L: number of layers
        keep_prob: probability that a node will be kept

    Returns:
        cache: dictionary containing the activations and dropout masks
    """
    cache = {}
    cache["A0"] = X

    for i in range(1, L + 1):
        W = weights["W{}".format(i)]
        b = weights["b{}".format(i)]

        A_prev = cache["A{}".format(i - 1)]
        Z = np.matmul(W, A_prev) + b

        if i == L:
            # Softmax output layer
            t = np.exp(Z)
            cache["A{}".format(i)] = t / np.sum(t, axis=0, keepdims=True)
        else:
            # Hidden layer (tanh)
            A = np.tanh(Z)

            # Dropout mask
            D = np.random.binomial(1, keep_prob, size=A.shape)

            # Apply inverted dropout
            A = (A * D) / keep_prob

            cache["D{}".format(i)] = D
            cache["A{}".format(i)] = A

    return cache
