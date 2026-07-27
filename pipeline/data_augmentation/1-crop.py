#!/usr/bin/env python3
"""
Module that performs a random crop on an image
"""

import tensorflow as tf


def crop_image(image, size):
    """
    Performs a random crop on an image

    Args:
        image: A 3D tf.Tensor containing the image
        size: A tuple containing the desired crop size
              (height, width, channels)

    Returns:
        The randomly cropped image
    """
    return tf.image.random_crop(image, size)
