#!/usr/bin/env python3
"""Performs convolution on images using multiple kernels."""

import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """
    Performs a convolution on images using multiple kernels.

    Args:
        images: numpy.ndarray (m, h, w, c)
        kernels: numpy.ndarray (kh, kw, c, nc)
        padding: 'same', 'valid', or tuple (ph, pw)
        stride: tuple (sh, sw)

    Returns:
        numpy.ndarray containing convolved images
    """
    m, h, w, c = images.shape
    kh, kw, _, nc = kernels.shape
    sh, sw = stride

    if padding == 'same':
        ph = int(np.ceil(((h - 1) * sh + kh - h) / 2))
        pw = int(np.ceil(((w - 1) * sw + kw - w) / 2))
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph, pw = padding

    padded = np.pad(
        images,
        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode='constant'
    )

    output_h = ((h + 2 * ph - kh) // sh) + 1
    output_w = ((w + 2 * pw - kw) // sw) + 1

    output = np.zeros((m, output_h, output_w, nc))

    # Three nested loops: output height, output width, and number of kernels
    for i in range(output_h):
        for j in range(output_w):
            for k in range(nc):
                output[:, i, j, k] = np.sum(
                    padded[
                        :,
                        i * sh:i * sh + kh,
                        j * sw:j * sw + kw,
                        :
                    ] * kernels[:, :, :, k],
                    axis=(1, 2, 3)
                )

    return output
