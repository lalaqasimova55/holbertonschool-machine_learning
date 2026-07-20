#!/usr/bin/env python3
"""Calculates the weighted moving average of a data set."""


def moving_average(data, beta):
    """
    Calculates the weighted moving average of a data set.

    Args:
        data (list): List of data values.
        beta (float): Weight used for the moving average.

    Returns:
        list: The bias-corrected moving averages.
    """
    moving_averages = []
    v = 0

    for i, value in enumerate(data, start=1):
        v = beta * v + (1 - beta) * value
        v_corrected = v / (1 - beta ** i)
        moving_averages.append(v_corrected)

    return moving_averages
