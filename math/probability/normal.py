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
            
            # Mean calculation
            self.mean = float(sum(data) / len(data))
            
            # Variance calculation (Population variance: sum / n)
            diff_sq = [(x - self.mean) ** 2 for x in data]
            variance = sum(diff_sq) / len(data)
            self.stddev = float(variance ** 0.5)

    def z_score(self, x):
        """Calculates the z-score of a given x-value"""
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """Calculates the x-value of a given z-score"""
        return (z * self.stddev) + self.mean

    def pdf(self, x):
        """Calculates the PDF for a given x-value"""
        pi = 3.1415926536
        e = 2.7182818285
        
        exponent = -((x - self.mean) ** 2) / (2 * (self.stddev ** 2))
        denominator = self.stddev * ((2 * pi) ** 0.5)
        
        return (1 / denominator) * (e ** exponent)

    def cdf(self, x):
        """Calculates the CDF for a given x-value"""
        # We use the approximation for erf(x) from Abramowitz and Stegun
        # Erf argument is (x - mean) / (stddev * sqrt(2))
        value = (x - self.mean) / (self.stddev * (2 ** 0.5))
        
        # Constants
        e = 2.7182818285
        a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429
        p = 0.3275911
        
        sign = 1 if value >= 0 else -1
        v_abs = abs(value)
        
        t = 1 / (1 + p * v_abs)
        
        # erf approximation formula
        erf = 1 - (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t) * (e ** (-v_abs**2))
        
        # Final CDF formula: 0.5 * (1 + erf(value))
        return 0.5 * (1 + sign * erf)
