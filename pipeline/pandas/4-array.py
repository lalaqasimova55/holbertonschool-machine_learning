#!/usr/bin/env python3
"""
Defines a function to extract the last 10 rows of specific columns as an array
"""


def array(df):
    """
    Selects the last 10 rows of the 'High' and 'Close' columns from a DataFrame
    and converts them into a numpy.ndarray without importing numpy.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        numpy.ndarray: A 10x2 NumPy array containing the selected values.
    """
    ndarray = df[['High', 'Close']].tail(10).to_numpy()
    return ndarray
