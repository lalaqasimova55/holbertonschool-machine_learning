#!/usr/bin/env python3
"""Performs back propagation over a convolutional layer."""

import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    Performs back propagation over a convolutional layer.

    Args:
        dZ: numpy.ndarray (m, h_new, w_new, c_new) of output derivatives
        A_prev: numpy.ndarray (m, h_prev, w_prev, c_prev) of previous layer
        W: numpy.ndarray (kh, kw, c_prev, c_new) containing the kernels
        b: numpy.ndarray (1, 1, 1, c_new) containing the biases
        padding: string, either "same" or "valid"
        stride: tuple (sh, sw) containing the strides

    Returns:
        dA_prev, dW, db respectively
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, c_new = W.shape
    sh, sw = stride
    _, h_new, w_new, _ = dZ.shape

    if padding == 'same':
        ph = int(np.ceil(((h_prev - 1) * sh + kh - h_prev) / 2))
        pw = int(np.ceil(((w_prev - 1) * sw + kw - w_prev) / 2))
    elif padding == 'valid':
        ph, pw = 0, 0

    padded_A = np.pad(
        A_prev,
        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode='constant'
    )

    dA_prev_padded = np.zeros_like(padded_A)
    dW = np.zeros_like(W)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    for i in range(h_new):
        for j in range(w_new):
            for k in range(c_new):
                h_start = i * sh
                h_end = h_start + kh
                w_start = j * sw
                w_end = w_start + kw

                a_slice = padded_A[:, h_start:h_end, w_start:w_end, :]

                # Extract and expand value to keep lines short
                dz_val = dZ[:, i, j, k, np.newaxis, np.newaxis, np.newaxis]

                # Accumulate gradients
                dA_prev_padded[:, h_start:h_end, w_start:w_end, :] += (
                    W[..., k] * dz_val
                )
                dW[..., k] += np.sum(a_slice * dz_val, axis=0)

    # Slice out the padding to get the actual dA_prev
    if ph > 0 and pw > 0:
        dA_prev = dA_prev_padded[:, ph:-ph, pw:-pw, :]
    elif ph > 0:
        dA_prev = dA_prev_padded[:, ph:-ph, :, :]
    elif pw > 0:
        dA_prev = dA_prev_padded[:, :, pw:-pw, :]
    else:
        dA_prev = dA_prev_padded

    return dA_prev, dW, db
