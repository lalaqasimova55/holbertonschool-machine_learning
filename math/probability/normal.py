#!/usr/bin/env python3
"""Normal distribution module"""

import math


class Normal:
    """Represents a normal distribution"""

    def __init__(self, data=None, mean=0., stddev=1.):
        """Initialize Normal distribution"""

        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)

        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")

            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            mean_val = sum(data) / len(data)

            variance = sum((x - mean_val) ** 2 for x in data) / len(data)
            stddev_val = variance ** 0.5

            self.mean = float(mean_val)
            self.stddev = float(stddev_val)

    def z_score(self, x):
        """Calculates the z-score of a given x-value"""
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """Calculates the x-value of a given z-score"""
        return (z * self.stddev) + self.mean

    def pdf(self, x):
        """Calculates the PDF for a given x-value"""

        pi = math.pi

        exponent = -((x - self.mean) ** 2) / (2 * (self.stddev ** 2))
        denominator = self.stddev * math.sqrt(2 * pi)

        return (1 / denominator) * math.exp(exponent)

    def cdf(self, x):
        """Calculates the CDF for a given x-value"""

        z = (x - self.mean) / (self.stddev * math.sqrt(2))

        return 0.5 * (1 + math.erf(z))
