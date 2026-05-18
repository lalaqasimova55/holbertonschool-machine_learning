#!/usr/bin/env python3
"""
Defines a function that extracts last 10 rows of High and Close columns
as a numpy.ndarray
"""


def array(df):
    """
    Returns last 10 rows of High and Close columns as numpy.ndarray
    """
    return df[['High', 'Close']].tail(10).to_numpy()
