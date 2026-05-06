#!/usr/bin/env python3
"""Normal distribution module"""


class Normal:
    """Class that represents a normal distribution"""

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

            self.mean = float(sum(data) / len(data))

            variance = sum((x - self.mean) ** 2 for x in data) / len(data)
            self.stddev = float(variance ** 0.5)

    def cdf(self, x):
        """
        Calculates the CDF for a given x-value

        Args:
            x (float): x-value

        Returns:
            float: CDF value
        """

        pi = 3.1415926536

        z = (x - self.mean) / (self.stddev * (2 ** 0.5))

        def erf(z):
            """Approximates the error function"""
            return (2 / (pi ** 0.5)) * (
                z
                - (z ** 3) / 3
                + (z ** 5) / 10
                - (z ** 7) / 42
                + (z ** 9) / 216
            )

        return 0.5 * (1 + erf(z))
