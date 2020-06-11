"""
Name: Functions for converting data
Developer: Matt Clarke
Date: Jan 1, 2020
Description: Allows you to convert the data types in of Pandas DataFrame columns.
"""


import pandas as pd


def cols_to_slugify(df, columns):
    """Slugify selected column values and return DataFrame.

    Args:
        df: Pandas DataFrame.
        columns: List of columns to convert.

    Returns:
        Original DataFrame with converted column data.
    """

    for col in columns:
        df[col] = df[col].str.lower().replace('[^0-9a-zA-Z]+', '_', regex=True)

    return df


def cols_to_float(df, columns):
    """Convert selected column values to float and return DataFrame.

    Args:
        df: Pandas DataFrame.
        columns: List of columns to convert.

    Returns:
        Original DataFrame with converted column data.
    """

    for col in columns:
        df[col] = df[col].astype(float)

    return df


def cols_to_int(df, columns):
    """Convert selected column values to int and return DataFrame.

    Args:
        df: Pandas DataFrame.
        columns: List of columns to convert.

    Returns:
        Original DataFrame with converted column data.
    """

    for col in columns:
        df[col] = df[col].astype(int)

    return df


def cols_to_datetime(df, columns):
    """Convert selected column values to datetime and return DataFrame.

    Args:
        df: Pandas DataFrame.
        columns: List of columns to convert.

    Returns:
        Original DataFrame with converted column data.
    """

    for col in columns:
        df[col] = pd.to_datetime(df[col])

    return df


def cols_to_negative(df, columns):
    """Convert selected column values to negative and return DataFrame.

    Args:
        df: Pandas DataFrame.
        columns: List of columns to convert.

    Returns:
        Original DataFrame with converted column data.
    """

    for col in columns:
        df[col] = df[col] * -1

    return df

