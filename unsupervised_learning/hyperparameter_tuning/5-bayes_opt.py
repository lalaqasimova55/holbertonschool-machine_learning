#!/usr/bin/env python3
"""Bayesian Optimization Module"""


import numpy as np
from scipy.stats import norm


GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """Performs Bayesian optimization on a noiseless 1D GP."""

    def __init__(
        self,
        f,
        X_init,
        Y_init,
        bounds,
        ac_samples,
        l=1,
        sigma_f=1,
        xsi=0.01,
        minimize=True
    ):
        """Initialize the Bayesian optimization."""
        self.f = f
        self.gp = GP(X_init, Y_init, l=l, sigma_f=sigma_f)
        self.X_s = np.linspace(
            bounds[0],
            bounds[1],
            ac_samples
        ).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """Calculate the next best sample location."""
        mu, sigma = self.gp.predict(self.X_s)

        if self.minimize:
            best = np.min(self.gp.Y)
            Z = (best - mu - self.xsi) / sigma
            EI = (
                (best - mu - self.xsi) * norm.cdf(Z)
                + sigma * norm.pdf(Z)
            )
        else:
            best = np.max(self.gp.Y)
            Z = (mu - best - self.xsi) / sigma
            EI = (
                (mu - best - self.xsi) * norm.cdf(Z)
                + sigma * norm.pdf(Z)
            )

        EI[sigma == 0] = 0

        X_next = self.X_s[np.argmax(EI)]

        return X_next, EI

    def optimize(self, iterations=100):
        """Optimize the black-box function."""
        for _ in range(iterations):
            X_next, _ = self.acquisition()

            if np.any(np.all(np.isclose(self.gp.X, X_next), axis=1)):
                break

            Y_next = self.f(X_next)
            self.gp.update(X_next, Y_next)

        if self.minimize:
            index = np.argmin(self.gp.Y)
        else:
            index = np.argmax(self.gp.Y)

        return self.gp.X[index], self.gp.Y[index]
