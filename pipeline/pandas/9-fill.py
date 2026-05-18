#!/usr/bin/env python3
"""
Defines a function that fills missing values in a DataFrame
"""


def fill(df):
    """
    Cleans and fills missing values in the dataframe
    """

    # Remove Weighted_Price column
    df = df.drop(columns=['Weighted_Price'])

    # Fill Close NaN with previous value
    df['Close'] = df['Close'].fillna(method='ffill')

    # Fill Open, High, Low with Close
    df['Open'] = df['Open'].fillna(df['Close'])
    df['High'] = df['High'].fillna(df['Close'])
    df['Low'] = df['Low'].fillna(df['Close'])

    # Fill volume columns with 0
    df['Volume_(BTC)'] = df['Volume_(BTC)'].fillna(0)
    df['Volume_(Currency)'] = df['Volume_(Currency)'].fillna(0)

    return df
