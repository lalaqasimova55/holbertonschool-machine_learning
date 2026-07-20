#!/usr/bin/env python3
"""
Same convolution on grayscale images
"""

import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    Performs a same convolution on grayscale images

    images: numpy.ndarray of shape (m, h, w)
    kernel: numpy.ndarray of shape (kh, kw)

    Returns:
        numpy.ndarray containing convolved images
    """

    m, h, w = images.shape
    kh, kw = kernel.shape

    # Padding miktarı
    pad_h = kh // 2
    pad_w = kw // 2

    # Zero padding
    padded = np.pad(
        images,
        ((0, 0), (pad_h, pad_h), (pad_w, pad_w)),
        mode='constant'
    )

    output = np.zeros((m, h, w))

    # Sadece 2 loop kullanılıyor
    for i in range(h):
        for j in range(w):
            output[:, i, j] = np.sum(
                padded[:, i:i + kh, j:j + kw] * kernel,
                axis=(1, 2)
            )

    return output
