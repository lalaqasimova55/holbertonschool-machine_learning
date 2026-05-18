#!/usr/bin/env python3
"""
Defines a function to rename a column, convert timestamps, and filter columns
"""
import pandas as pd


def rename(df):
    """
    Renames the column Timestamp to Datetime, converts its values to datetime
    formats, and filters the DataFrame to display only Datetime and Close.

    Args:
        df (pd.DataFrame): The input DataFrame containing a 'Timestamp' column.

    Returns:
        pd.DataFrame: The modified DataFrame with only 'Datetime' and 'Close'.
    """
    # 1. Rename the 'Timestamp' column to 'Datetime'
    df = df.rename(columns={'Timestamp': 'Datetime'})

    # 2. Convert the timestamp values (in seconds) to datetime objects
    df['Datetime'] = pd.to_datetime(df['Datetime'], unit='s')

    # 3. Filter the DataFrame to include only 'Datetime' and 'Close' columns
    df = df[['Datetime', 'Close']]

    return df
