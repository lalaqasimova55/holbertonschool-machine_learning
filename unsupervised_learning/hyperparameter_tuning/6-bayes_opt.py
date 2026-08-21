#!/usr/bin/env python3
"""Bayesian optimization of a neural network."""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import GPyOpt

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l2


np.random.seed(42)
tf.random.set_seed(42)


# ---------------------------------------------------------
# Example dataset
# ---------------------------------------------------------

X = np.random.randn(2000, 10)

true_w = np.random.randn(10, 1)
Y = X @ true_w + np.random.randn(2000, 1) * 0.1

split = int(0.8 * len(X))

X_train = X[:split]
Y_train = Y[:split]

X_val = X[split:]
Y_val = Y[split:]


# ---------------------------------------------------------
# Model
# ---------------------------------------------------------

def build_model(learning_rate, units, dropout, l2_value):
    """Build and compile the neural network."""

    model = Sequential([
        Dense(
            units,
            activation='relu',
            input_shape=(X_train.shape[1],),
            kernel_regularizer=l2(l2_value)
        ),
        Dropout(dropout),
        Dense(
            units,
            activation='relu',
            kernel_regularizer=l2(l2_value)
        ),
        Dense(1)
    ])

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate
    )

    model.compile(
        optimizer=optimizer,
        loss='mae',
        metrics=['mae']
    )

    return model


# ---------------------------------------------------------
# Objective function
# ---------------------------------------------------------

def objective(params):
    """Train the model and return validation MAE."""

    learning_rate = float(params[0][0])
    units = int(params[0][1])
    dropout = float(params[0][2])
    l2_value = float(params[0][3])
    batch_size = int(params[0][4])

    checkpoint = (
        f'checkpoint_lr{learning_rate:.6f}'
        f'_units{units}'
        f'_dropout{dropout:.3f}'
        f'_l2{l2_value:.8f}'
        f'_batch{batch_size}.keras'
    )

    model = build_model(
        learning_rate,
        units,
        dropout,
        l2_value
    )

    early_stopping = EarlyStopping(
        monitor='val_mae',
        patience=10,
        restore_best_weights=True
    )

    model_checkpoint = ModelCheckpoint(
        checkpoint,
        monitor='val_mae',
        save_best_only=True,
        mode='min'
    )

    history = model.fit(
        X_train,
        Y_train,
        validation_data=(X_val, Y_val),
        epochs=100,
        batch_size=batch_size,
        callbacks=[
            early_stopping,
            model_checkpoint
        ],
        verbose=0
    )

    best_mae = min(history.history['val_mae'])

    return np.array([[best_mae]])
