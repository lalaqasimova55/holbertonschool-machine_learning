#!/usr/bin/env python3
"""
Defines a function that sorts a DataFrame by High price in descending order
"""


def high(df):
    """
    Sorts the DataFrame by the High column in descending order

    Args:
        df (pd.DataFrame): input dataframe

    Returns:
        pd.DataFrame: sorted dataframe
    """
    return df.sort_values(by='High', ascending=False)
