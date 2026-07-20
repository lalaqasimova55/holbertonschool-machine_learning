#!/usr/bin/env python3
"""Performs forward propagation over a convolutional layer."""

import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    Performs forward propagation over a convolutional layer of a neural network.

    Args:
        A_prev: numpy.ndarray (m, h_prev, w_prev, c_prev) - previous layer output
        W: numpy.ndarray (kh, kw, c_prev, c_new) - kernels
        b: numpy.ndarray (1, 1, 1, c_new) - biases
        activation: activation function applied to the convolution
        padding: string, either "same" or "valid"
        stride: tuple (sh, sw) containing the strides

    Returns:
        The output of the convolutional layer
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, c_new = W.shape
    sh, sw = stride

    if padding == 'same':
        ph = int(np.ceil(((h_prev - 1) * sh + kh - h_prev) / 2))
        pw = int(np.ceil(((w_prev - 1) * sw + kw - w_prev) / 2))
    elif padding == 'valid':
        ph, pw = 0, 0

    padded = np.pad(
        A_prev,
        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode='constant'
    )

    output_h = ((h_prev + 2 * ph - kh) // sh) + 1
    output_w = ((w_prev + 2 * pw - kw) // sw) + 1

    # Initialize intermediate Z matrix before activation
    Z = np.zeros((m, output_h, output_w, c_new))

    # Three nested loops over spatial outputs and target output channels
    for i in range(output_h):
        for j in range(output_w):
            for k in range(c_new):
                Z[:, i, j, k] = np.sum(
                    padded[
                        :,
                        i * sh:i * sh + kh,
                        j * sw:j * sw + kw,
                        :
                    ] * W[:, :, :, k],
                    axis=(1, 2, 3)
                )

    # Add the bias term (broadcasting handles batch and spatial dimensions)
    Z = Z + b

    # Apply the activation function
    return activation(Z)
