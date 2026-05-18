#!/usr/bin/env python3
"""
Defines a function that concatenates two DataFrames with keys
"""

import pandas as pd
index = __import__('10-index').index


def concat(df1, df2):
    """
    Index both DataFrames on Timestamp and concatenate them

    Args:
        df1 (pd.DataFrame): coinbase dataframe
        df2 (pd.DataFrame): bitstamp dataframe

    Returns:
        pd.DataFrame: concatenated dataframe
    """

    df1 = index(df1)
    df2 = index(df2)

    df2 = df2[df2.index <= 1417411920]

    return pd.concat([df2, df1], keys=['bitstamp', 'coinbase'])
