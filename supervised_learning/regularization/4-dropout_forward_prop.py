#!/usr/bin/env python3
"""Forward propagation with Dropout"""

import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Conducts forward propagation using Dropout.

    Args:
        X: numpy.ndarray of shape (nx, m)
        weights: dictionary containing the weights and biases
        L: number of layers
        keep_prob: probability of keeping a neuron active

    Returns:
        Dictionary containing:
            A0 ... AL : activations
            D1 ... D(L-1) : dropout masks
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
            exp = np.exp(Z - np.max(Z, axis=0, keepdims=True))
            A = exp / np.sum(exp, axis=0, keepdims=True)
            cache["A{}".format(i)] = A
        else:
            # Hidden layers (tanh)
            A = np.tanh(Z)

            # Dropout mask
            D = np.random.rand(*A.shape) < keep_prob
            A = A * D
            A = A / keep_prob

            cache["D{}".format(i)] = D
            cache["A{}".format(i)] = A

    return cache
