#!/usr/bin/env python3
"""
Trains a Keras model
"""


def train_model(network, data, labels, batch_size, epochs,
                verbose=True, shuffle=False):
    """
    Trains a model using mini-batch gradient descent.

    Args:
        network: the Keras model to train
        data: numpy.ndarray of shape (m, nx) containing the input data
        labels: one-hot numpy.ndarray of shape (m, classes)
        batch_size: size of each mini-batch
        epochs: number of epochs to train
        verbose: whether to print training progress
        shuffle: whether to shuffle the data each epoch

    Returns:
        The History object generated after training
    """
    history = network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        shuffle=shuffle
    )

    return history
