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
