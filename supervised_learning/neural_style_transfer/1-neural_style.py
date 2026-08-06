#!/usr/bin/env python3
"""
Neural Style Transfer module with VGG19 model loading.
"""
import numpy as np
import tensorflow as tf


class NST:
    """
    NST class performs tasks for neural style transfer.
    """
    style_layers = [
        'block1_conv1',
        'block2_conv1',
        'block3_conv1',
        'block4_conv1',
        'block5_conv1'
    ]
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
        Class constructor for Neural Style Transfer.

        Args:
            style_image: numpy.ndarray of shape (h, w, 3) - style reference
            content_image: numpy.ndarray of shape (h, w, 3) - content reference
            alpha: weight for content cost
            beta: weight for style cost
        """
        if not isinstance(style_image, np.ndarray) or \
           style_image.ndim != 3 or style_image.shape[2] != 3:
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)"
            )

        if not isinstance(content_image, np.ndarray) or \
           content_image.ndim != 3 or content_image.shape[2] != 3:
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)"
            )

        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")

        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        self.load_model()

    @staticmethod
    def scale_image(image):
        """
        Rescales an image such that its pixel values are between 0 and 1
        and its largest side is 512 pixels.

        Args:
            image: numpy.ndarray of shape (h, w, 3) containing the image

        Returns:
            The scaled image as a tf.Tensor of shape (1, h_new, w_new, 3)
        """
        if not isinstance(image, np.ndarray) or \
           image.ndim != 3 or image.shape[2] != 3:
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)"
            )

        h, w, _ = image.shape

        if h > w:
            h_new = 512
            w_new = int(w * (512 / h))
        else:
            w_new = 512
            h_new = int(h * (512 / w))

        image_expanded = tf.expand_dims(image, axis=0)

        resized_image = tf.image.resize(
            image_expanded,
            size=[h_new, w_new],
            method=tf.image.ResizeMethod.BICUBIC
        )

        scaled_image = tf.clip_by_value(resized_image / 255.0, 0.0, 1.0)

        return scaled_image

    def load_model(self):
        """
        Creates the model used to calculate style and content cost.

        Uses VGG19 pretrained on ImageNet as base. Replaces MaxPooling layers
        with AveragePooling layers for smoother style transfer results.
        """
        vgg = tf.keras.applications.VGG19(
            include_top=False,
            weights='imagenet'
        )

        x = vgg.input
        layer_outputs = {}

        for layer in vgg.layers[1:]:
            if isinstance(layer, tf.keras.layers.MaxPooling2D):
                x = tf.keras.layers.AveragePooling2D(
                    pool_size=layer.pool_size,
                    strides=layer.strides,
                    padding=layer.padding,
                    name=layer.name
                )(x)
            else:
                x = layer(x)
            layer_outputs[layer.name] = x

        outputs = [
            layer_outputs[layer]
            for layer in self.style_layers + [self.content_layer]
        ]

        # Explicitly set name="model" to guarantee the model summary name
        model = tf.keras.Model(inputs=vgg.input, outputs=outputs, name="model")
        model.trainable = False
        self.model = model
