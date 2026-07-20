#!/usr/bin/env python3
"""Performs back propagation over a pooling layer."""

import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs back propagation over a pooling layer.

    Args:
        dA: numpy.ndarray (m, h_new, w_new, c_new) of output derivatives
        A_prev: numpy.ndarray (m, h_prev, w_prev, c) of previous layer
        kernel_shape: tuple of (kh, kw) containing the kernel size
        stride: tuple of (sh, sw) containing the pooling strides
        mode: string, either 'max' or 'avg'

    Returns:
        The partial derivatives with respect to the previous layer (dA_prev)
    """
    m, h_prev, w_prev, c = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride
    _, h_new, w_new, _ = dA.shape

    dA_prev = np.zeros_like(A_prev)

    # Two loops over the spatial coordinates of the output grid
    for i in range(h_new):
        for j in range(w_new):
            h_start = i * sh
            h_end = h_start + kh
            w_start = j * sw
            w_end = w_start + kw

            slice_window = A_prev[:, h_start:h_end, w_start:w_end, :]

            if mode == 'max':
                # Create a binary mask where the max value matches the element
                # Keep spatial dimensions intact to compare element-wise
                max_val = np.max(slice_window, axis=(1, 2), keepdims=True)
                mask = (slice_window == max_val)

                # Reshape dA value for proper broadcasting over the window
                da_val = dA[:, i, j, :, np.newaxis, np.newaxis]
                da_val = np.transpose(da_val, (0, 2, 3, 1))

                dA_prev[:, h_start:h_end, w_start:w_end, :] += mask * da_val

            elif mode == 'avg':
                # Distribute the gradient evenly across the window area
                da_val = dA[:, i, j, :, np.newaxis, np.newaxis]
                da_val = np.transpose(da_val, (0, 2, 3, 1))
                distributed_da = da_val / (kh * kw)

                dA_prev[:, h_start:h_end, w_start:w_end, :] += distributed_da

    return dA_prev
