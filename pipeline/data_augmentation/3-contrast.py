#!/usr/bin/env python3
"""
Module that randomly adjusts the contrast of an image
"""

import tensorflow as tf


def change_contrast(image, lower, upper):
    """
    Randomly adjusts the contrast of an image.

    Args:
        image: A 3D tf.Tensor containing the image.
        lower: Lower bound for the random contrast factor.
        upper: Upper bound for the random contrast factor.

    Returns:
        The contrast-adjusted image.
    """
    return tf.image.random_contrast(image, lower=lower, upper=upper)
