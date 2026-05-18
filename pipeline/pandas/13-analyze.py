#!/usr/bin/env python3
"""
Defines a function that computes descriptive statistics
for a DataFrame excluding Timestamp column
"""

def analyze(df):
    """
    Computes descriptive statistics for all columns except Timestamp

    Args:
        df (pd.DataFrame): input dataframe

    Returns:
        pd.DataFrame: statistics dataframe
    """
    return df.drop(columns=['Timestamp']).describe()
