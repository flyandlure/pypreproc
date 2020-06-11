"""
Name: Functions for correcting data
Developer: Matt Clarke
Date: Jan 1, 2020
Description: Allows you to correct data in of Pandas DataFrame columns and column names.
"""


def cols_to_strip_characters(df, columns, character):
    """Strip a given character from selected DataFrame columns.

    Args:
        df: Pandas DataFrame.
        columns: List of columns to convert.
        character: Character to strip
    Returns:
        Original DataFrame with converted column data.
    """

    for col in columns:
        df[col] = df[col].str.replace(character, '')

    return df


def cols_to_drop(df, columns):
    """Drop selected columns and return DataFrame.

    Args:
        df: Pandas DataFrame.
        columns: List of columns to drop.

    Returns:
        Original DataFrame without dropped columns.
    """

    for col in columns:
        df.drop([col], axis=1, inplace=True)

    return df


def col_names_to_lower(df):
    """Convert column names to lowercase and return DataFrame.

    Args:
        df: Pandas DataFrame.

    Returns:
        Original DataFrame with converted column data.
    """

    return df.columns.str.lower()


def cols_to_rename(df, dictionary):
    """Drop selected columns and return DataFrame.

    Args:
        df: Pandas DataFrame.
        dictionary: {'old_name1': 'new_name1', 'old_name2': 'new_name2'}

    Returns:
        Original DataFrame without dropped columns.
    """

    df.rename(dictionary, axis=1, inplace=True)

    return df

