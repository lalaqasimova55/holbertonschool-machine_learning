#!/usr/bin/env python3
"""
ResNet-50 architecture
"""

from tensorflow import keras as K

identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """
    Builds the ResNet-50 architecture

    Returns:
        A Keras model
    """
    initializer = K.initializers.he_normal(seed=0)

    X = K.Input(shape=(224, 224, 3))

    # Initial layers
    Y = K.layers.Conv2D(
        filters=64,
        kernel_size=(7, 7),
        strides=(2, 2),
        padding='same',
        kernel_initializer=initializer
    )(X)
    Y = K.layers.BatchNormalization(axis=3)(Y)
    Y = K.layers.Activation('relu')(Y)
    Y = K.layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2),
        padding='same'
    )(Y)

    # Stage 2
    Y = projection_block(Y, [64, 64, 256], s=1)
    Y = identity_block(Y, [64, 64, 256])
    Y = identity_block(Y, [64, 64, 256])

    # Stage 3
    Y = projection_block(Y, [128, 128, 512], s=2)
    Y = identity_block(Y, [128, 128, 512])
    Y = identity_block(Y, [128, 128, 512])
    Y = identity_block(Y, [128, 128, 512])

    # Stage 4
    Y = projection_block(Y, [256, 256, 1024], s=2)
    Y = identity_block(Y, [256, 256, 1024])
    Y = identity_block(Y, [256, 256, 1024])
    Y = identity_block(Y, [256, 256, 1024])
    Y = identity_block(Y, [256, 256, 1024])
    Y = identity_block(Y, [256, 256, 1024])

    # Stage 5
    Y = projection_block(Y, [512, 512, 2048], s=2)
    Y = identity_block(Y, [512, 512, 2048])
    Y = identity_block(Y, [512, 512, 2048])

    # Average Pooling
    Y = K.layers.AveragePooling2D(
        pool_size=(7, 7),
        strides=(1, 1)
    )(Y)

    # Output layer
    Y = K.layers.Dense(
        units=1000,
        activation='softmax',
        kernel_initializer=initializer
    )(Y)

    model = K.models.Model(inputs=X, outputs=Y)

    return model
