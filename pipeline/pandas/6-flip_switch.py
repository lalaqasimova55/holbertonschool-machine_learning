#!/usr/bin/env python3
"""
Defines a function that flips and switches a DataFrame
"""


def flip_switch(df):
    """
    Sorts data in reverse chronological order and transposes it
    """
    return df.sort_index(ascending=False).T
