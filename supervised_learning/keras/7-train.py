#!/usr/bin/env python3
"""
Trains a Keras model using mini-batch gradient descent
with optional validation, early stopping, and learning
rate decay.
"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False,
                alpha=0.1, decay_rate=1,
                verbose=True, shuffle=False):
    """
    Trains a model.

    Args:
        network: Keras model
        data: input data
        labels: one-hot labels
        batch_size: mini-batch size
        epochs: number of epochs
        validation_data: validation data
        early_stopping: whether to use early stopping
        patience: patience for early stopping
        learning_rate_decay: whether to use learning rate decay
        alpha: initial learning rate
        decay_rate: decay rate
        verbose: verbosity mode
        shuffle: whether to shuffle the data

    Returns:
        The History object.
    """
    callbacks = []

    if validation_data is not None:
        if early_stopping:
            callbacks.append(
                K.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=patience
                )
            )

        if learning_rate_decay:
            def schedule(epoch):
                """Inverse time decay schedule."""
                return alpha / (1 + decay_rate * epoch)

            callbacks.append(
                K.callbacks.LearningRateScheduler(
                    schedule,
                    verbose=1
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
