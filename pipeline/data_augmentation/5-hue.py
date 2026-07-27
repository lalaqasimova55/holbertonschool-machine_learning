#!/usr/bin/env python3
"""
Module that changes the hue of an image
"""

import tensorflow as tf


def change_hue(image, delta):
    """
    Changes the hue of an image.

    Args:
        image: A 3D tf.Tensor containing the image.
        delta: The amount to change the hue.

    Returns:
        The hue-adjusted image.
    """
    return tf.image.adjust_hue(image, delta)
