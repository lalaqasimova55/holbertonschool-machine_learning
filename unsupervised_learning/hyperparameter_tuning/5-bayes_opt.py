#!/usr/bin/env python3
"""Bayesian optimization module"""
import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """Performs Bayesian optimization on a noiseless 1D Gaussian process"""

    def __init__(self, f, X_init, Y_init, bounds, ac_samples, l=1,
                 sigma_f=1, xsi=0.01, minimize=True):
        """
        Class constructor

        f: the black-box function to be optimized
        X_init: numpy.ndarray of shape (t, 1) representing the inputs
            already sampled with the black-box function
        Y_init: numpy.ndarray of shape (t, 1) representing the outputs
            of the black-box function for each input in X_init
        t: the number of initial samples
        bounds: tuple of (min, max) representing the bounds of the
            space in which to look for the optimal point
        ac_samples: the number of samples that should be analyzed
            during acquisition
        l: the length parameter for the kernel
        sigma_f: the standard deviation given to the output of the
            black-box function
        xsi: the exploration-exploitation factor for acquisition
        minimize: a bool determining whether optimization should be
            performed for minimization (True) or maximization (False)
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        min_bound, max_bound = bounds
        X_s = np.linspace(min_bound, max_bound, ac_samples)
        self.X_s = X_s.reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """
        Calculates the next best sample location using Expected
        Improvement

        Returns: X_next, EI
            X_next: numpy.ndarray of shape (1,) representing the next
                best sample point
            EI: numpy.ndarray of shape (ac_samples,) containing the
                expected improvement of each potential sample
        """
        mu, sigma = self.gp.predict(self.X_s)

        if self.minimize:
            Y_sample_opt = np.min(self.gp.Y)
            imp = Y_sample_opt - mu - self.xsi
        else:
            Y_sample_opt = np.max(self.gp.Y)
            imp = mu - Y_sample_opt - self.xsi

        with np.errstate(divide='warn'):
            Z = np.zeros_like(sigma)
            mask = sigma > 0
            Z[mask] = imp[mask] / sigma[mask]
            EI = np.zeros_like(sigma)
            term1 = imp[mask] * norm.cdf(Z[mask])
            term2 = sigma[mask] * norm.pdf(Z[mask])
            EI[mask] = term1 + term2

        X_next = self.X_s[np.argmax(EI)]

        return X_next, EI

    def optimize(self, iterations=100):
        """
        Optimizes the black-box function

        iterations: the maximum number of iterations to perform

        If the next proposed point is one that has already been
        sampled, optimization should be stopped early

        Returns: X_opt, Y_opt
            X_opt: numpy.ndarray of shape (1,) representing the
                optimal point
            Y_opt: numpy.ndarray of shape (1,) representing the
                optimal function value
        """
        for _ in range(iterations):
            X_next, _ = self.acquisition()

            if np.any(np.isclose(X_next, self.gp.X)):
                break

            Y_next = self.f(X_next)

            self.gp.update(X_next, Y_next)

        if self.minimize:
            idx = np.argmin(self.gp.Y)
        else:
            idx = np.argmax(self.gp.Y)

        X_opt = self.gp.X[idx]
        Y_opt = self.gp.Y[idx]

        return X_opt, Y_opt
