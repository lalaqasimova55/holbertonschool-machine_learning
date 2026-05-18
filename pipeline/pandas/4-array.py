#!/usr/bin/env python3
"""
Defines a function to extract the last 10 rows of specific columns as an array
"""

def array(df):
    """
    Selects the last 10 rows of the 'High' and 'Close' columns from a DataFrame
    and returns them as a list of lists.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        list: A 10x2 list containing the selected values.
    """
    return df[['High', 'Close']].tail(10).values.tolist()
