#!/usr/bin/env python3
"""
Performs convolution on grayscale images
"""

import numpy as np


def convolve_grayscale(images, kernel, padding='same',
                       stride=(1, 1)):
    """
    Performs a convolution on grayscale images

    Args:
        images: numpy.ndarray (m, h, w)
        kernel: numpy.ndarray (kh, kw)
        padding: 'same', 'valid' or tuple(ph, pw)
        stride: tuple(sh, sw)

    Returns:
        numpy.ndarray containing convolved images
    """

    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    if padding == 'same':
        ph = ((h - 1) * sh + kh - h) // 2
        pw = ((w - 1) * sw + kw - w) // 2

    elif padding == 'valid':
        ph = 0
        pw = 0

    elif isinstance(padding, tuple):
        ph, pw = padding

    else:
        raise ValueError("Invalid padding")

    images_pad = np.pad(
        images,
        ((0, 0), (ph, ph), (pw, pw)),
        mode='constant'
    )

    new_h = ((h + 2 * ph - kh) // sh) + 1
    new_w = ((w + 2 * pw - kw) // sw) + 1

    output = np.zeros((m, new_h, new_w))

    for i in range(new_h):
        for j in range(new_w):

            region = images_pad[
                :,
                i * sh:i * sh + kh,
                j * sw:j * sw + kw
            ]

            output[:, i, j] = np.sum(
                region * kernel,
                axis=(1, 2)
            )

    return output
