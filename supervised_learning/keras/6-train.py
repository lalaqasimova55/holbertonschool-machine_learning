#!/usr/bin/env python3
"""
Trains a Keras model using mini-batch gradient descent
with optional validation and early stopping.
"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, verbose=True, shuffle=False):
    """
    Trains a model.

    Args:
        network: the Keras model to train
        data: input data
        labels: one-hot labels
        batch_size: mini-batch size
        epochs: number of epochs
        validation_data: data used for validation
        early_stopping: whether to use early stopping
        patience: patience for early stopping
        verbose: verbosity mode
        shuffle: whether to shuffle the data

    Returns:
        The History object generated after training.
    """
    callbacks = []

    if early_stopping and validation_data is not None:
        callbacks.append(
            K.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience
            )
        )

    history = network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=validation_data,
        callbacks=callbacks,
        verbose=verbose,
        shuffle=shuffle
    )

    return history
