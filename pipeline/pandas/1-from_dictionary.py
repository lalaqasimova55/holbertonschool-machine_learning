#!/usr/bin/env python3
"""
Creates a pandas DataFrame from a dictionary and saves it in a variable `df`
"""
import pandas as pd

# Dictionary containing the specified column data
data = {
    'First': [0.0, 0.5, 1.0, 1.5],
    'Second': ['one', 'two', 'three', 'four']
}

# Creating the DataFrame with the designated row labels (indices)
df = pd.DataFrame(data, index=['A', 'B', 'C', 'D'])
