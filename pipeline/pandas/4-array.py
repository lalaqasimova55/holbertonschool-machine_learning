#!/usr/bin/env python3
"""
Defines a function to extract the last 10 rows of specific columns as a NumPy array
"""
import numpy as np


def array(df):
    """
    Selects the last 10 rows of the 'High' and 'Close' columns from a DataFrame
    and converts them into a numpy.ndarray.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        np.ndarray: A 10x2 NumPy array containing the selected values.
    """
    # Select the High and Close columns, grab the last 10 rows, and convert to numpy
    ndarray = df[['High', 'Close']].tail(10).to_numpy()
    
    return ndarray
