"""
Name: Helper functions
Developer: Matt Clarke
Date: Jan 1, 2020
Description: Helper functions for accessing Pandas data.
"""

from sklearn import feature_selection


def select(df, column_name, operator, where=[], exclude=False):
    """Select DataFrame rows based on a column value using a
    range of operators, including startswith, endswith, isin
    and contains with include and exclude filters.
    ----------
    :param df: Pandas dataframe
    :param column_name: Column name to search within
    :param operator: Operator (startswith, endswith, contains, or isin)
    :param exclude: Optional parameter to include or exclude based on where
    ---------
    Examples:
    ends = select(df, 'item_code', 'endswith', where='A4', exclude=False)
    sw = select(df, 'item_code', 'startswith', where='WPG', exclude=False)
    contains = select(df, 'item_code', 'contains', where='A4', exclude=False)
    isin = select(df, 'item_code', 'isin', where=['PFG3A4','PFG3A5'], exclude=False)
    _is = select(df, 'code', 'is', where='B', exclude=False)
    """

    if operator == 'startswith':
        selected = df[column_name].str.startswith(where)
    elif operator == 'endswith':
        selected = df[column_name].str.endswith(where)
    elif operator == 'contains':
        selected = df[column_name].str.contains(where)
    elif operator == 'isin':
        selected = df[column_name].isin(where)
    elif operator == 'is':
        selected = df[column_name] == where
    else:
        selected = df[column_name].str.contains(where)

    return df[selected] if not exclude else df[~selected]


def get_unique_rows(df, columns, sort_by):
    """De-dupe a Pandas DataFrame and return unique rows.

    :param df: Pandas DataFrame.
    :param columns: List of columns to return.
    :param sort_by: Column to sort by.
    :return: Pandas DataFrame de-duped to remove duplicate rows.
    """

    df = df[columns]
    df = df.drop_duplicates(subset=None, keep='last', inplace=False)
    df = df.sort_values(sort_by, ascending=False)
    return df


def get_low_var_cols(df, threshold):
    """Analyse a Pandas DataFrame, extract the numeric columns and
    return a list of those which have a variance below the threshold.

    Args:
        :param df: Pandas DataFrame.
        :param threshold: Variance threshold.

    Returns:
        List of columns with variance below the threshold.

    Example:
        low_var_cols = get_low_var_cols(df, 0.01)

    """
    df = df.select_dtypes(['number'])
    selector = feature_selection.VarianceThreshold(threshold=threshold)
    selector.fit(df)
    return df.columns[~selector.get_support()]

