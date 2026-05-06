#!/usr/bin/env python3
"""Module for Normal distribution"""


class Normal:
    """Class that represents a normal distribution"""

    def __init__(self, data=None, mean=0., stddev=1.):
        """Initialize the distribution"""
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
            
            # Calculate mean of the data set
            self.mean = float(sum(data) / len(data))
            
            # Calculate variance and standard deviation
            variance = sum((x - self.mean) ** 2 for x in data) / len(data)
            self.stddev = float(variance ** 0.5)

    def z_score(self, x):
        """Calculates the z-score of a given x-value"""
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """Calculates the x-value of a given z-score"""
        return (z * self.stddev) + self.mean

    def pdf(self, x):
        """Calculates the value of the PDF for a given x-value"""
        pi = 3.1415926536
        e = 2.7182818285
        
        exponent = -((x - self.mean) ** 2) / (2 * (self.stddev ** 2))
        denominator = self.stddev * ((2 * pi) ** 0.5)
        
        return (1 / denominator) * (e ** exponent)

    def cdf(self, x):
        """Calculates the value of the CDF for a given x-value"""
        # Constants for calculations
        e = 2.7182818285
        
        # Value used for the error function (erf)
        # Formula: 0.5 * (1 + erf( (x - mean) / (stddev * sqrt(2)) ))
        value = (x - self.mean) / (self.stddev * (2 ** 0.5))
        
        # Abramowitz & Stegun approximation constants for erf
        a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429
        p = 0.3275911
        
        # Handle the sign for the approximation
        sign = 1 if value >= 0 else -1
        v_abs = abs(value)
        
        # t calculation
        t = 1.0 / (1.0 + p * v_abs)
        
        # Polynomial approximation
        poly = (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t)
        
        # erf(|value|)
        erf_abs = 1.0 - poly * (e ** (-(v_abs**2)))
        
        # Correct erf(value) with original sign
        erf_final = sign * erf_abs
        
        return 0.5 * (1 + erf_final)
