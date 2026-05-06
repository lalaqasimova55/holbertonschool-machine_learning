#!/usr/bin/env python3
"""Normal distribution module"""


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

            # mean
            mean_val = sum(data) / len(data)

            # variance
            variance = sum((x - mean_val) ** 2 for x in data) / len(data)

            # stddev
            stddev_val = variance ** 0.5

            self.mean = float(mean_val)
            self.stddev = float(stddev_val)
