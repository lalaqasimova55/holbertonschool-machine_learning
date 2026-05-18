#!/usr/bin/env python3
"""
Defines a function to load data from a file into a pandas DataFrame
"""
import pandas as pd


def from_file(filename, delimiter):
    """
    Loads data from a file as a pd.DataFrame.

    Args:
        filename (str): The path to the file to load.
        delimiter (str): The column separator character.

    Returns:
        pd.DataFrame: The loaded DataFrame content.
    """
    df = pd.read_csv(filename, sep=delimiter)
    return df
