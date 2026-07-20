#!/usr/bin/env python3
"""
Strided convolution on grayscale images
"""

import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on grayscale images

    images: numpy.ndarray of shape (m, h, w)
    kernel: numpy.ndarray of shape (kh, kw)
    padding: 'same', 'valid', or tuple (ph, pw)
    stride: tuple (sh, sw)

    Returns:
        numpy.ndarray containing convolved images
    """

    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    # Determine padding
    if padding == 'same':
        ph = int(((h - 1) * sh + kh - h) / 2)
        pw = int(((w - 1) * sw + kw - w) / 2)

    elif padding == 'valid':
        ph = 0
        pw = 0

    elif isinstance(padding, tuple):
        ph, pw = padding

    else:
        raise ValueError("Invalid padding")

    # Apply zero padding
    padded = np.pad(
        images,
        ((0, 0), (ph, ph), (pw, pw)),
        mode='constant'
    )

    # Output size
    output_h = int(
        ((h + 2 * ph - kh) / sh) + 1
    )

    output_w = int(
        ((w + 2 * pw - kw) / sw) + 1
    )

    output = np.zeros((m, output_h, output_w))

    # Only two loops allowed
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
