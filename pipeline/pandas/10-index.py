#!/usr/bin/env python3
"""
Defines a function that sets Timestamp as index of DataFrame
"""


def index(df):
    """
    Sets the Timestamp column as the index

    Args:
        df (pd.DataFrame): input dataframe

    Returns:
        pd.DataFrame: modified dataframe
    """
    return df.set_index('Timestamp')
