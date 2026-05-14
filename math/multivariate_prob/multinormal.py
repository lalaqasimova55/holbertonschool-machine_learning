#!/usr/bin/env python3
"""Multivariate Normal distribution class"""

import numpy as np


class MultiNormal:
    """Represents a Multivariate Normal distribution"""

    def __init__(self, data):
        """
        Class constructor

        Args:
            data: numpy.ndarray of shape (d, n)
        """

        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise TypeError("data must be a 2D numpy.ndarray")

        d, n = data.shape

        if n < 2:
            raise ValueError("data must contain multiple data points")

        # mean (d, 1)
        self.mean = np.mean(data, axis=1, keepdims=True)

        # center data
        X = data - self.mean

        # covariance (d, d)
        self.cov = (X @ X.T) / (n - 1)
