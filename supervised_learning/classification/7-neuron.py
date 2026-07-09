#!/usr/bin/env python3
"""Neuron class for binary classification."""

import numpy as np
import matplotlib.pyplot as plt


class Neuron:
    """Defines a single neuron performing binary classification."""

    def __init__(self, nx):
        """
        Class constructor.

        Args:
            nx (int): Number of input features.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Getter for W."""
        return self.__W

    @property
    def b(self):
        """Getter for b."""
        return self.__b

    @property
    def A(self):
        """Getter for A."""
        return self.__A

    def forward_prop(self, X):
        """
        Calculates forward propagation.

        Args:
            X (numpy.ndarray): Input data.

        Returns:
            numpy.ndarray: Activated output.
        """
        Z = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A

    def cost(self, Y, A):
        """
        Calculates the cost using logistic regression.

        Args:
            Y (numpy.ndarray): Correct labels.
            A (numpy.ndarray): Activated output.

        Returns:
            float: Cost.
        """
        m = Y.shape[1]
        return -np.sum(
            Y * np.log(A) +
            (1 - Y) * np.log(1.0000001 - A)
        ) / m

    def evaluate(self, X, Y):
        """
        Evaluates the neuron's predictions.

        Args:
            X (numpy.ndarray): Input data.
            Y (numpy.ndarray): Correct labels.

        Returns:
            tuple: (prediction, cost)
        """
        A = self.forward_prop(X)
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, self.cost(Y, A)

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Calculates one pass of gradient descent.

        Args:
            X (numpy.ndarray): Input data.
            Y (numpy.ndarray): Correct labels.
            A (numpy.ndarray): Activated output.
            alpha (float): Learning rate.
        """
        m = Y.shape[1]

        dZ = A - Y
        dW = np.matmul(dZ, X.T) / m
        db = np.sum(dZ) / m

        self.__W -= alpha * dW
        self.__b -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """
        Trains the neuron.

        Args:
            X (numpy.ndarray): Input data.
            Y (numpy.ndarray): Correct labels.
            iterations (int): Number of iterations.
            alpha (float): Learning rate.
            verbose (bool): Print training progress.
            graph (bool): Plot training cost.
            step (int): Step size for printing/graphing.

        Returns:
            tuple: (prediction, cost)
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")

        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        if verbose or graph:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError(
                    "step must be positive and <= iterations"
                )

        costs = []
        steps = []

        for i in range(iterations + 1):
            A = self.forward_prop(X)

            if i % step == 0 or i == iterations:
                cost = self.cost(Y, A)

                if verbose:
                    print("Cost after {} iterations: {}".format(i, cost))

                if graph:
                    steps.append(i)
                    costs.append(cost)

            if i < iterations:
                self.gradient_descent(X, Y, A, alpha)

        if graph:
            plt.plot(steps, costs, 'b')
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.show()

        return self.evaluate(X, Y)
