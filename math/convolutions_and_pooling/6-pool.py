#!/usr/bin/env python3
"""Performs pooling on images."""

import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    Performs pooling on images.

    Args:
        images: numpy.ndarray (m, h, w, c) containing multiple images
        kernel_shape: tuple of (kh, kw) containing the kernel shape
        stride: tuple of (sh, sw) containing the strides
        mode: indicates the type of pooling ('max' or 'avg')

    Returns:
        numpy.ndarray containing the pooled images
    """
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    output_h = ((h - kh) // sh) + 1
    output_w = ((w - kw) // sw) + 1

    output = np.zeros((m, output_h, output_w, c))

    # Exactly two loops over the spatial dimensions of the output
    for i in range(output_h):
        for j in range(output_w):
            slice_window = images[
                :,
                i * sh:i * sh + kh,
                j * sw:j * sw + kw,
                :
            ]
            if mode == 'max':
                output[:, i, j, :] = np.max(slice_window, axis=(1, 2))
            elif mode == 'avg':
                output[:, i, j, :] = np.mean(slice_window, axis=(1, 2))

    return output
