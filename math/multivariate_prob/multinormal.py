#!/usr/bin/env python3
"""Multivariate Normal distribution class"""

import numpy as np


class MultiNormal:
    """Represents a Multivariate Normal distribution"""

    def __init__(self, data):
        """
        Class constructor

        data: numpy.ndarray of shape (d, n)
        """

        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise TypeError("data must be a 2D numpy.ndarray")

        d, n = data.shape

        if n < 2:
            raise ValueError("data must contain multiple data points")

        # mean: (d, 1)
        self.mean = np.mean(data, axis=1, keepdims=True)

        # covariance: (d, d)
        X = data - self.mean
        self.cov = (X @ X.T) / (n - 1)

    def pdf(self, x):
        """
        Calculates the PDF at a data point x
        """

        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")

        d = self.mean.shape[0]

        if x.shape != (d, 1):
            raise ValueError(f"x must have the shape ({d}, 1)")

        diff = x - self.mean

        cov_inv = np.linalg.inv(self.cov)
        det_cov = np.linalg.det(self.cov)

        exponent = -0.5 * (diff.T @ cov_inv @ diff)

        norm_const = 1 / (np.sqrt(((2 * np.pi) ** d) * det_cov))

        return float(norm_const * np.exp(exponent))
