#!/usr/bin/env python3
"""
Defines a function that removes rows with NaN in Close column
"""


def prune(df):
    """
    Removes entries where Close has NaN values

    Args:
        df (pd.DataFrame): input dataframe

    Returns:
        pd.DataFrame: cleaned dataframe
    """
    return df.dropna(subset=['Close'])
