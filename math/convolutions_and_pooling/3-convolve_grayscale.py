#!/usr/bin/env python3
"""Performs convolution on grayscale images."""

import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on grayscale images.

    Args:
        images: numpy.ndarray (m, h, w)
        kernel: numpy.ndarray (kh, kw)
        padding: 'same', 'valid', or tuple
        stride: tuple (sh, sw)

    Returns:
        numpy.ndarray containing convolved images
    """

    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    if padding == 'same':
        # Use np.ceil to ensure odd numbers round up properly
        ph = int(np.ceil(((h - 1) * sh + kh - h) / 2))
        pw = int(np.ceil(((w - 1) * sw + kw - w) / 2))

    elif padding == 'valid':
        ph = 0
        pw = 0

    else:
        ph, pw = padding

    padded = np.pad(
        images,
        ((0, 0), (ph, ph), (pw, pw)),
        mode='constant'
    )

    output_h = ((h + 2 * ph - kh) // sh) + 1
    output_w = ((w + 2 * pw - kw) // sw) + 1

    output = np.zeros((m, output_h, output_w))

    for i in range(output_h):
        for j in range(output_w):
            output[:, i, j] = np.sum(
                padded[
                    :,
                    i * sh:i * sh + kh,
                    j * sw:j * sw + kw
                ] * kernel,
                axis=(1, 2)
            )

    return output
