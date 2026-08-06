#!/usr/bin/env python3
"""
Neural Style Transfer module with total style cost calculation.
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
        self.generate_features()

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

        model = tf.keras.Model(inputs=vgg.input, outputs=outputs, name="model")
        model.trainable = False
        self.model = model

    @staticmethod
    def gram_matrix(input_layer):
        """
        Calculates the Gram matrix of a given layer output tensor.

        Args:
            input_layer: instance of tf.Tensor or tf.Variable of shape
                         (1, h, w, c) containing the layer output

        Returns:
            tf.Tensor of shape (1, c, c) containing the gram matrix
        """
        if not isinstance(input_layer, (tf.Tensor, tf.Variable)) or \
           len(input_layer.shape) != 4:
            raise TypeError("input_layer must be a tensor of rank 4")

        gram = tf.linalg.einsum('bijc,bijd->bcd', input_layer, input_layer)

        input_shape = tf.shape(input_layer)
        num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)

        return gram / num_locations

    def generate_features(self):
        """
        Extracts features used to calculate neural style cost.

        Sets public instance attributes:
            gram_style_features: list of Gram matrices for style layers
            content_feature: content layer output tensor for content image
        """
        style_preprocessed = tf.keras.applications.vgg19.preprocess_input(
            self.style_image * 255.0
        )
        content_preprocessed = tf.keras.applications.vgg19.preprocess_input(
            self.content_image * 255.0
        )

        style_outputs = self.model(style_preprocessed)
        content_outputs = self.model(content_preprocessed)

        self.gram_style_features = [
            self.gram_matrix(style_outputs[i])
            for i in range(len(self.style_layers))
        ]

        self.content_feature = content_outputs[-1]

    def layer_style_cost(self, style_output, gram_target):
        """
        Calculates the style cost for a single layer.

        Args:
            style_output: tf.Tensor or tf.Variable of shape (1, h, w, c)
                          containing the layer style output of generated image
            gram_target: tf.Tensor or tf.Variable of shape (1, c, c)
                         containing the target Gram matrix for that layer

        Returns:
            tf.Tensor containing the layer's style cost
        """
        if not isinstance(style_output, (tf.Tensor, tf.Variable)) or \
           len(style_output.shape) != 4:
            raise TypeError("style_output must be a tensor of rank 4")

        c = style_output.shape[-1]

        if not isinstance(gram_target, (tf.Tensor, tf.Variable)) or \
           gram_target.shape != (1, c, c):
            raise TypeError(
                f"gram_target must be a tensor of shape [1, {c}, {c}]"
            )

        gram_style = self.gram_matrix(style_output)

        return tf.reduce_mean(tf.square(gram_style - gram_target))

    def style_cost(self, style_outputs):
        """
        Calculates the total style cost for the generated image.

        Args:
            style_outputs: list of tf.Tensor style outputs for generated image

        Returns:
            tf.Tensor containing the total style cost
        """
        num_layers = len(self.style_layers)

        if not isinstance(style_outputs, list) or \
           len(style_outputs) != num_layers:
            raise TypeError(
                f"style_outputs must be a list with a length of {num_layers}"
            )

        weight = 1.0 / num_layers
        style_cost = 0.0

        for i in range(num_layers):
            layer_cost = self.layer_style_cost(
                style_outputs[i],
                self.gram_style_features[i]
            )
            style_cost += weight * layer_cost

        return style_cost
