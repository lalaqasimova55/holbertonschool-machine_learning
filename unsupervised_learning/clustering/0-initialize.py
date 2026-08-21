#!/usr/bin/env python3
"""Gaussian Process"""


import numpy as np


class GaussianProcess:
    """Represents a noiseless 1D Gaussian process."""

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """Initialize the Gaussian process."""
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """Calculate the covariance kernel matrix."""
        return (self.sigma_f ** 2) * np.exp(
            -((X1 - X2.T) ** 2) / (2 * self.l ** 2)
        )

    def predict(self, X_s):
        """Predict the mean and variance of points in the GP."""
        K = self.K
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s)

        K_inv = np.linalg.inv(K)

        mu = K_s.T @ K_inv @ self.Y
        sigma = np.diag(K_ss - K_s.T @ K_inv @ K_s)

        return mu.flatten(), sigma
