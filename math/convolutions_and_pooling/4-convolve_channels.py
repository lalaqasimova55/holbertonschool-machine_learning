#!/usr/bin/env python3
"""Performs convolution on images with channels."""

import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on images with channels.

    Args:
        images: numpy.ndarray (m, h, w, c)
        kernel: numpy.ndarray (kh, kw, c)
        padding: 'same', 'valid', or tuple (ph, pw)
        stride: tuple (sh, sw)

    Returns:
        numpy.ndarray containing convolved images
    """
    m, h, w, c = images.shape
    kh, kw, _ = kernel.shape
    sh, sw = stride

    if padding == 'same':
        ph = int(np.ceil(((h - 1) * sh + kh - h) / 2))
        pw = int(np.ceil(((w - 1) * sw + kw - w) / 2))
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph, pw = padding

    # Pad only the height and width axes (axes 1 and 2)
    padded = np.pad(
        images,
        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode='constant'
    )

    output_h = ((h + 2 * ph - kh) // sh) + 1
    output_w = ((w + 2 * pw - kw) // sw) + 1

    output = np.zeros((m, output_h, output_w))

    # Exactly two loops over the spatial dimension of the output matrix
    for i in range(output_h):
        for j in range(output_w):
            output[:, i, j] = np.sum(
                padded[
                    :,
                    i * sh:i * sh + kh,
                    j * sw:j * sw + kw,
                    :
                ] * kernel,
                axis=(1, 2, 3)
            )

    return output
