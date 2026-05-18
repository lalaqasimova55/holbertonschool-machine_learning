#!/usr/bin/env python3
"""
Defines a function that creates a hierarchical concatenation of two DataFrames
"""

import pandas as pd
index = __import__('10-index').index


def hierarchy(df1, df2):
    """
    Rearranges MultiIndex so Timestamp is first level and
    concatenates filtered data from both DataFrames

    Args:
        df1 (pd.DataFrame): coinbase dataframe
        df2 (pd.DataFrame): bitstamp dataframe

    Returns:
        pd.DataFrame: hierarchical concatenated dataframe
    """

    df1 = index(df1)
    df2 = index(df2)

    # filter required range
    df2 = df2[(df2.index >= 1417411980) & (df2.index <= 1417417980)]
    df1 = df1[(df1.index >= 1417411980) & (df1.index <= 1417417980)]

    # concat with keys
    df = pd.concat([df2, df1], keys=['bitstamp', 'coinbase'])

    # ensure Timestamp is first level
    df = df.swaplevel(0, 1)

    # sort by Timestamp (chronological order)
    df = df.sort_index(level=0)

    return df
