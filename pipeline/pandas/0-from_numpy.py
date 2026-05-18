#!/usr/bin/env python3
"""
Defines a function that creates a pd.DataFrame from a np.ndarray
"""
import pandas as pd


def from_numpy(array):
    """
    Creates a pd.DataFrame from a np.ndarray with alphabetical column labels.

    Args:
        array: np.ndarray from which to create the DataFrame

    Returns:
        The newly created pd.DataFrame
    """
    num_cols = array.shape[1]
    columns = [chr(65 + i) for i in range(num_cols)]
    df = pd.DataFrame(array, columns=columns)
    return df
