#!/usr/bin/env python3
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Updates the weights and biases of a neural network with dropout
    using gradient descent
    """

    m = Y.shape[1]
    dz = {}

    # Output layer
    A = cache['A' + str(L)]
    dz['dz' + str(L)] = A - Y

    for layer in range(L, 0, -1):
        A_prev = cache['A' + str(layer - 1)]

        dW = np.matmul(dz['dz' + str(layer)],
                       A_prev.T) / m

        db = np.sum(dz['dz' + str(layer)],
                    axis=1,
                    keepdims=True) / m

        weights['W' + str(layer)] -= alpha * dW
        weights['b' + str(layer)] -= alpha * db

        if layer > 1:
            W = weights['W' + str(layer)]

            dz_prev = np.matmul(W.T,
                                dz['dz' + str(layer)])

            A_prev = cache['A' + str(layer - 1)]

            dz_prev = dz_prev * (1 - np.square(A_prev))

            # Apply dropout mask
            D = cache['D' + str(layer - 1)]
            dz_prev *= D
            dz_prev /= keep_prob

            dz['dz' + str(layer - 1)] = dz_prev
