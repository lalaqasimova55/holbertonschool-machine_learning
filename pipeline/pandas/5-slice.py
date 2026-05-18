#!/usr/bin/env python3
"""
Defines a function to slice specific columns and rows from a pandas DataFrame
"""


def slice(df):
    """
    Extracts the columns High, Low, Close, and Volume_(BTC),
    and selects every 60th row.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The sliced DataFrame view.
    """
    columns = ['High', 'Low', 'Close', 'Volume_(BTC)']
    sliced_df = df[columns].iloc[::60]
    return sliced_df
