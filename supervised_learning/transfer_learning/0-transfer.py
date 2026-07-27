#!/usr/bin/env python3
"""
Transfer Learning with CIFAR-10
"""

from tensorflow import keras as K
import tensorflow as tf
import numpy as np


def preprocess_data(X, Y):
    """
    Preprocesses the CIFAR-10 dataset

    Args:
        X: numpy.ndarray of shape (m, 32, 32, 3)
        Y: numpy.ndarray of shape (m,)

    Returns:
        X_p, Y_p
    """
    X_p = K.applications.efficientnet.preprocess_input(
        X.astype("float32")
    )

    Y_p = K.utils.to_categorical(Y, 10)

    return X_p, Y_p


if __name__ == "__main__":

    (X_train, Y_train), (X_test, Y_test) = \
        K.datasets.cifar10.load_data()

    X_train, Y_train = preprocess_data(X_train, Y_train)
    X_test, Y_test = preprocess_data(X_test, Y_test)

    inputs = K.Input(shape=(32, 32, 3))

    # Resize CIFAR-10 images
    x = K.layers.Lambda(
        lambda image: tf.image.resize(image, (224, 224))
    )(inputs)

    # Data augmentation
    x = K.layers.RandomFlip("horizontal")(x)
    x = K.layers.RandomRotation(0.1)(x)
    x = K.layers.RandomZoom(0.1)(x)

    base_model = K.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_tensor=x,
        pooling="avg"
    )

    base_model.trainable = False

    # Classification head
    x = K.layers.Dropout(0.3)(base_model.output)
    outputs = K.layers.Dense(
        10,
        activation="softmax",
        kernel_initializer="he_normal"
    )(x)

    model = K.models.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=K.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks = [
        K.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True
        ),
        K.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=2,
            min_lr=1e-6
        )
    ]

    history = model.fit(
        X_train,
        Y_train,
        batch_size=64,
        epochs=20,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )

    # Fine-tuning
    base_model.trainable = True

    # Freeze all except the last 20 layers
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    model.compile(
        optimizer=K.optimizers.Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Fine-tune the model
    model.fit(
        X_train,
        Y_train,
        batch_size=64,
        epochs=10,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate on the test set
    loss, accuracy = model.evaluate(
        X_test,
        Y_test,
        batch_size=128,
        verbose=1
    )

    print("Test accuracy:", accuracy)

    # Save the trained model
    model.save("cifar10.h5")
