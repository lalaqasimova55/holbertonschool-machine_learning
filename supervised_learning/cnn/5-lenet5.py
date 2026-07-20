#!/usr/bin/env python3
"""Builds a modified version of the LeNet-5 architecture using tensorflow."""

import tensorflow.compat.v1 as tf


def lenet5(x, y):
    """
    Builds a modified LeNet-5 architecture using tensorflow 1.x layers.

    Args:
        x: tf.compat.v1.placeholder (m, 28, 28, 1) containing the input images
        y: tf.compat.v1.placeholder (m, 10) containing the one-hot labels

    Returns:
        y_pred: tensor containing the activated output
        train_op: training operation utilizing Adam optimizer
        loss: tensor containing the cross-entropy loss
    """
    # He normal initializer for all weight-bearing layers
    init = tf.compat.v1.keras.initializers.VarianceScaling(
        scale=2.0,
        mode='fan_in',
        distribution='normal'
    )

    # Layer 1: Conv 5x5, 6 filters, same padding, relu activation
    conv1 = tf.compat.v1.layers.conv2d(
        inputs=x,
        filters=6,
        kernel_size=5,
        padding='same',
        activation=tf.nn.relu,
        kernel_initializer=init
    )

    # Layer 2: Max pool 2x2, stride 2x2
    pool1 = tf.compat.v1.layers.max_pooling2d(
        inputs=conv1,
        pool_size=2,
        strides=2
    )

    # Layer 3: Conv 5x5, 16 filters, valid padding, relu activation
    conv2 = tf.compat.v1.layers.conv2d(
        inputs=pool1,
        filters=16,
        kernel_size=5,
        padding='valid',
        activation=tf.nn.relu,
        kernel_initializer=init
    )

    # Layer 4: Max pool 2x2, stride 2x2
    pool2 = tf.compat.v1.layers.max_pooling2d(
        inputs=conv2,
        pool_size=2,
        strides=2
    )

    # Flatten layer
    flat = tf.compat.v1.layers.flatten(inputs=pool2)

    # Layer 5: Fully connected layer with 120 nodes, relu activation
    fc1 = tf.compat.v1.layers.dense(
        inputs=flat,
        units=120,
        activation=tf.nn.relu,
        kernel_initializer=init
    )

    # Layer 6: Fully connected layer with 84 nodes, relu activation
    fc2 = tf.compat.v1.layers.dense(
        inputs=fc1,
        units=84,
        activation=tf.nn.relu,
        kernel_initializer=init
    )

    # Layer 7: Fully connected output layer (logits) with 10 nodes
    logits = tf.compat.v1.layers.dense(
        inputs=fc2,
        units=10,
        kernel_initializer=init
    )

    # Softmax activated output
    y_pred = tf.nn.softmax(logits)

    # Cross entropy loss calculation
    loss = tf.compat.v1.losses.softmax_cross_entropy(
        onehot_labels=y,
        logits=logits
    )

    # Adam Optimizer training operation
    optimizer = tf.compat.v1.train.AdamOptimizer()
    train_op = optimizer.minimize(loss)

    return y_pred, train_op, loss
