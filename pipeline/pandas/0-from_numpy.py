#!/usr/bin/env python3
"""
Defines a function that creates a pd.DataFrame from a np.ndarray
"""
import pandas as pd


def from_numpy(array):
    """
    Creates a pd.DataFrame from a np.ndarray with alphabetical column labels.

    Args:
        array: np.ndarray from which to create the DataFrame

    Returns:
        The newly created pd.DataFrame
    ```
    # Sütunların sayını massivin formasına (shape) əsasən tapırıq
    num_cols = array.shape[1]

    # ASCII dəyərlərindən istifadə edərək 'A'-dan başlayaraq lazımi sayda hərf generasiya edirik
    # 'A' hərfinin ASCII kodu 65-dir
    columns = [chr(65 + i) for i in range(num_cols)]

    # DataFrame-i yaradırıq və sütun adlarını təyin edirik
    df = pd.DataFrame(array, columns=columns)

    return df
