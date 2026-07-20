#!/usr/bin/env python3
"""Builds a modified version of the LeNet-5 architecture using keras."""

from tensorflow import keras as K


def lenet5(X):
    """
    Builds a modified LeNet-5 architecture using keras.

    Args:
        X: K.Input of shape (m, 28, 28, 1) containing the input images

    Returns:
        A K.Model compiled to use Adam optimization and accuracy metrics
    """
    # Layer 1: Conv 5x5, 6 filters, same padding, relu activation
    conv1 = K.layers.Conv2D(
        filters=6,
        kernel_size=(5, 5),
        padding='same',
        activation='relu',
        kernel_initializer=K.initializers.he_normal(seed=0)
    )(X)

    # Layer 2: Max pool 2x2, stride 2x2
    pool1 = K.layers.MaxPooling2D(
        pool_size=(2/2, 2),
        strides=(2, 2)
    )(conv1)

    # Layer 3: Conv 5x5, 16 filters, valid padding, relu activation
    conv2 = K.layers.Conv2D(
        filters=16,
        kernel_size=(5, 5),
        padding='valid',
        activation='relu',
        kernel_initializer=K.initializers.he_normal(seed=0)
    )(pool1)

    # Layer 4: Max pool 2x2, stride 2x2
    pool2 = K.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)
    )(conv2)

    # Flatten the pool layer output to match the input of Dense layers
    flat = K.layers.Flatten()(pool2)

    # Layer 5: Fully connected layer with 120 nodes, relu activation
    fc1 = K.layers.Dense(
        units=120,
        activation='relu',
        kernel_initializer=K.initializers.he_normal(seed=0)
    )(flat)

    # Layer 6: Fully connected layer with 84 nodes, relu activation
    fc2 = K.layers.Dense(
        units=84,
        activation='relu',
        kernel_initializer=K.initializers.he_normal(seed=0)
    )(fc1)

    # Layer 7: Fully connected output layer with 10 nodes, softmax activation
    output = K.layers.Dense(
        units=10,
        activation='softmax',
        kernel_initializer=K.initializers.he_normal(seed=0)
    )(fc2)

    # Construct and compile the final model
    model = K.Model(inputs=X, outputs=output)
    model.compile(
        optimizer=K.optimizers.Adam(),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
