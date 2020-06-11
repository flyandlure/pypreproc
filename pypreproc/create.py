"""
Name: Functions for creating data
Developer: Matt Clarke
Date: Jan 1, 2020
Description: Functions to create new Pandas DataFrame features from existing data.
"""

import pandas as pd
import numpy as np
import itertools


def cols_to_log(df, columns):
    """Transform column data with log and return new columns of prefixed data.

    For us with data where the column values do not include zeroes.

    Args:
        df: Pandas DataFrame.
        columns: List of columns to transform.

    Returns:
        Original DataFrame with additional prefixed columns.
    """

    for col in columns:
        df['log_' + col] = np.log(df[col])

    return df


def cols_to_log1p(df, columns):
    """Transform column data with log+1 and return new columns of prefixed data.

    For use with data where the column values include zeroes.

    Args:
        df: Pandas DataFrame.
        columns: List of columns to transform.

    Returns:
        Original DataFrame with additional prefixed columns.
    """

    for col in columns:
        df['log1p_' + col] = np.log(df[col] + 1)

    return df


def cols_to_log_max_root(df, columns):
    """Convert data points to log values using the maximum value as the log max and return new columns of prefixed data.

    For use with data where the column values include zeroes.

    Args:
        df: Pandas DataFrame.
        columns: List of columns to transform.

    Returns:
        Original DataFrame with additional prefixed columns.
    """

    for col in columns:
        log_max = np.log(df[col].max())
        df['logmr_' + col] = df[col] ** (1 / log_max)

    return df


def cols_to_tanh(df, columns):
    """Transform column data with hyperbolic tangent and return new columns of prefixed data.

    Args:
        df: Pandas DataFrame.
        columns: List of columns to transform.

    Returns:
        Original DataFrame with additional prefixed columns.
    """

    for col in columns:
        df['tanh_' + col] = np.tanh(df[col])

    return df


def cols_to_sigmoid(df, columns):
    """Convert data points to values between 0 and 1 using a sigmoid function and return new columns of prefixed data.

    Args:
        df: Pandas DataFrame.
        columns: List of columns to transform.

    Returns:
        Original DataFrame with additional prefixed columns.
    """

    for col in columns:
        e = np.exp(1)
        y = 1 / (1 + e ** (-df[col]))
        df['sig_' + col] = y

    return df


def cols_to_cube_root(df, columns):
    """Convert data points to their cube root value so all values are between 0-1 and return new columns of prefixed data.

    Args:
        df: Pandas dataframe.
        columns: List of columns to transform.

    Returns:
        Original dataframe with additional prefixed columns.
    """

    for col in columns:
        df['cube_root_' + col] = df[col] ** (1 / 3)

    return df


def cols_to_cube_root_normalize(df, columns):
    """Convert data points to their normalized cube root value so all values are between 0-1 and return new columns of prefixed data.

    Args:
        df: Pandas DataFrame.
        columns: List of columns to transform.

    Returns:
        Original DataFrame with additional prefixed columns.
    """

    for col in columns:
        df['cube_root_' + col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min()) ** (1 / 3)

    return df


def cols_to_percentile(df, columns):
    """Convert data points to their percentile linearized value and return new columns of prefixed data.

    Args:
        df: Pandas DataFrame.
        columns: List of columns to transform.

    Returns:
        Original DataFrame with additional prefixed columns.
    """

    for col in columns:
        df['pc_lin_' + col] = df[col].rank(method='min').apply(lambda x: (x - 1) / len(df[col]) - 1)

    return df


def cols_to_normalize(df, columns):
    """Convert data points to values between 0 and 1 and return new columns of prefixed data.

    Args:
        df: Pandas DataFrame.
        columns: List of columns to transform.

    Returns:
        Original DataFrame with additional prefixed columns.
    """

    for col in columns:
        df['norm_' + col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

    return df


def cols_to_log1p_normalize(df, columns):
    """Transform column data with log+1 normalized and return new columns of prefixed data.

    For use with data where the column values include zeroes.

    Args:
        df: Pandas DataFrame.
        columns: List of columns to transform.

    Returns:
        Original DataFrame with additional prefixed columns.
    """

    for col in columns:
        df['log1p_norm_' + col] = np.log((df[col] - df[col].min()) / (df[col].max() - df[col].min()) + 1)

    return df


def cols_to_one_hot(df, columns):
    """One hot encode column values and return new prefixed columns.

    Args:
        df: Pandas DataFrame.
        columns: List of columns to transform.

    Returns:
        Original DataFrame with additional prefixed columns.
    """

    for col in columns:
        encoding = pd.get_dummies(df[col], prefix=col)
        df = pd.concat([df, encoding], axis=1)

    return df


def cols_to_reduce_uniques(df, column_threshold_dict):
    """Reduce the number of unique values by creating a column of X values and the rest marked "Others".

    Args:
        column_threshold_dict:
        df: Pandas DataFrame.
        columns: Dictionary of column and threshold, i.e. {'col1' : 1000, 'col2' : 3000}

    Returns:
        Original DataFrame with additional prefixed columns. The most dominant values in the column will
        be assigned their original value. The less dominant results will be assigned to Others, which can
        help visualise and model data in some cases.
    """

    for key, value in column_threshold_dict.items():
        counts = df[key].value_counts()
        others = set(counts[counts < value].index)
        df['reduce_' + key] = df[key].replace(list(others), 'Others')

    return df


def cols_to_count_uniques(df, group, columns):
    """Count the number of unique column values when grouping by another column and return new columns in original dataframe.

    Args:
        df: Pandas DataFrame.
        group: Column name to groupby
        columns: Columns to count uniques.

    Returns:
        Original DataFrame with new columns containing prefixed unique value counts.

    """

    for col in columns:
        df['unique_' + col + '_by_' + group] = df.groupby(group)[col].transform('nunique')
    return df


def cols_to_count(df, group, columns):
    """Count the number of column values when grouping by another column and return new columns in original dataframe.

    Args:
        df: Pandas DataFrame.
        group: Column name to groupby
        columns: Columns to count.

    Returns:
        Original DataFrame with new columns containing prefixed count values.

    """

    for col in columns:
        df['count_' + col + '_by_' + group] = df.groupby(group)[col].transform('count')
    return df


def cols_to_sum(df, group, columns):
    """Sum the values of a given column or columns based on a groupby parameter and return new columns in original DataFrame.

    Args:
        df: Pandas DataFrame.
        group: Column name to groupby
        columns: Columns to sum.

    Returns:
        Original DataFrame with new columns containing prefixed sum values.

    """

    for col in columns:
        df['sum_' + col + '_by_' + group] = df.groupby(group)[col].transform('sum')
    return df


def get_previous_value(df, group, column, name):
    """Group by a column and return the previous value of another column and assign value to a new column.

    Args:
        df: Pandas DataFrame.
        group: Column name to groupby
        column: Column value to return.
        name: Name for new column.

    Returns:
        Original DataFrame with new column containing previous value of named column.

    """
    df = df.copy()
    df[name] = df.groupby([group])[column].shift(-1)
    return df


def get_days_since_date(df, before_datetime, after_datetime, name):
    """Return a new column containing the difference between two dates in days.

    Args:
        df: Pandas DataFrame.
        before_datetime: Earliest datetime (will convert value)
        after_datetime: Latest datetime (will convert value)
        name: Name for new column.

    Returns:
        Original DataFrame with new column containing previous value of named column.

    """
    df = df.copy()
    df[before_datetime] = pd.to_datetime(df[before_datetime])
    df[after_datetime] = pd.to_datetime(df[after_datetime])

    df[name] = df[after_datetime] - df[before_datetime]
    df[name] = df[name] / np.timedelta64(1, 'D')

    return df


def get_dates(df, date_column):
    """Converts a given date to various formats and returns an updated DataFrame.

    Args:
        df: Pandas DataFrame.
        date_column:

    Returns:
        Original DataFrame with additional date columns.
    """

    df['day'] = df[date_column].dt.strftime("%d")  # Day of month with leading zero
    df['month'] = df[date_column].dt.strftime("%m")  # Month of year with leading zero
    df['year'] = df[date_column].dt.strftime("%Y")  # Full numeric four digit year
    df['year_month'] = df[date_column].dt.strftime("%Y%m")  # Full numeric four digit year plus month
    df['week_number'] = df[date_column].dt.strftime("%U")  # Week number with leading zero
    df['day_number'] = df[date_column].dt.strftime("%j")  # Day number with leading zero
    df['day_name'] = df[date_column].dt.strftime("%A")  # Day name, i.e. Sunday
    df['month_name'] = df[date_column].dt.strftime("%B")  # Month name, i.e. January
    df['mysql_date'] = df[date_column].dt.strftime("%Y-%d-%m")  # MySQL date, i.e. 2020-30-01
    df['quarter'] = df[date_column].dt.quarter  # Quarter with leading zero, i.e. 01

    return df


def get_grouped_stats(df, group, columns):
    """Group by a column and return summary statistics for a list of columns and add prefixed data to new columns.

    Args:
        df: Pandas DataFrame.
        group: Column name to groupby
        columns: Columns to summarise.

    Returns:
        Original DataFrame with new columns containing prefixed summary statistics.
        Example: customer_total_net_mean
    """

    for col in columns:
        df[group + '_' + col + '_mean'] = df.groupby([group])[col].transform('mean')
        df[group + '_' + col + '_median'] = df.groupby([group])[col].transform('median')
        df[group + '_' + col + '_std'] = df.groupby([group])[col].transform('std')
        df[group + '_' + col + '_max'] = df.groupby([group])[col].transform('max')
        df[group + '_' + col + '_min'] = df.groupby([group])[col].transform('min')

    return df


def get_feature_interactions(df, columns, depth):
    """Combine multiple features to create new features based on X unique combinations.

    Args:
        df: Pandas DataFrame.
        columns: Columns to combine in unique combinations.
        depth: Integer denoting the number of features to combine (2, 3 or 4)

    Returns:
        Original DataFrame with new columns containing prefixed features.

    """

    interactions = list(itertools.combinations(columns, r=depth))

    for interaction in interactions:
        name = '_'.join(interaction)

        if depth == 2:
            df[name] = df[interaction[0]] + df[interaction[1]]
        elif depth == 3:
            df[name] = df[interaction[0]] + df[interaction[1]] + df[interaction[2]]
        else:
            df[name] = df[interaction[0]] + df[interaction[1]] + df[interaction[2]] + df[interaction[3]]

    return df


def get_binned_data(df, column, name, bins):
    """Perform a simple quantile binning operation on a column and return a column of binned data in dataframe.

    Args:
        df: Pandas DataFrame.
        column: Column name to bin.
        name: Name for new column.
        bins: Number of bins to create.

    Returns:
        Original DataFrame with new column containing binned data.

    Usage:
        quotes = get_binned_data(quotes, 'customer_previous_quotes', 'customer_previous_quotes_bin', 5)
    """

    df = df.copy()
    df[name] = pd.qcut(df[column], q=bins, precision=0, labels=False, duplicates='drop')
    return df


def sum_columns(df, columns):
    """Get the sum of a list of columns.

        Args:
            df: Pandas DataFrame.
            columns: List of columns to sum.

        Returns:
            Value of summed columns.

        Usage:
            columns = ['quote_includes_sample', 'quote_free_shipping']
            df['sum'] = sum_columns(df, columns)
        """
    for col in columns:
        df[col] = df[col].astype(int)

    total = df[columns].astype(int).sum(axis=1)
    return total


def get_diff(df, column1, column2):
    """Get the difference between two column values.

        Args:
            df: Pandas DataFrame.
            column1: First column.
            column2: Second column.

        Returns:
            Value of summed columns.

        Usage:
            df['item_quantity_vs_mean'] = get_diff(df, 'item_quantity', 'item_code_item_quantity_mean')
        """
    return df[column1] - df[column2]


def get_previous_cumulative_sum(df, group, sum_column, sort_column):
    """Get the previous cumulative sum of a column based on a GroupBy.
    For example, calculate the running total of a customer's previous
    converted quotes, prior to the current one.

    Args:
        :param df: Pandas DataFrame.
        :param group: Column to group by.
        :param sum_column: Column to sum.
        :param sort_column: Column to sort by.

    Returns:
        Cumulative sum of the column minus the current row sum.

    Usage:
        df['running_total'] = get_previous_cumulative_sum(check, 'customer_id', 'quote_total_net', 'quote_date')

    """

    df = df.sort_values(by=sort_column, ascending=True)
    return df.groupby([group])[sum_column].cumsum() - df[sum_column]


def get_cumulative_sum(df, group, sum_column, sort_column):
    """Get the cumulative sum of a column based on a GroupBy.

    Args:
        :param df: Pandas DataFrame.
        :param group: Column to group by.
        :param sum_column: Column to sum.
        :param sort_column: Column to sort by.

    Returns:
        Cumulative sum of the column minus the current row sum.

    Usage:
        df['running_total'] = get_cumulative_sum(check, 'customer_id', 'quote_total_net', 'quote_date')

    """
    df = df.sort_values(by=sort_column, ascending=True)
    return df.groupby([group])[sum_column].cumsum()


def get_previous_cumulative_count(df, group, count_column, sort_column):
    """Get the previous cumulative count of a column based on a GroupBy.
    For example, calculate the running total of a customer's previous
    converted quotes, prior to the current one.

    Args:
        :param df: Pandas DataFrame.
        :param group: Column to group by.
        :param count_column: Column to count.
        :param sort_column: Column to sort by.

    Returns:
        Cumulative count of the column minus the current row sum.

    Usage:
        df['previous_quotes'] = get_previous_cumulative_count(check, 'customer_id', 'quote_id', 'quote_date')

    """

    df = df.sort_values(by=sort_column, ascending=True)
    return df.groupby([group])[count_column].cumcount() - 1


def get_rolling_average(df, group, column, periods, sortby):
    """Return the rolling average for a column based on a groupby.

    Args:
        :param df: Pandas DataFrame.
        :param group: Column to group by.
        :param column: Column to average.
        :param periods: Number of periods to use.
        :param sortby: Column to sort by.

    Returns:
        Conversion rate.
    """

    df = df.sort_values(by=[sortby], ascending=True)
    df = df.groupby(group)[column].apply(lambda x: x.rolling(center=False, window=periods).mean())
    return df


def get_probability_ratio(df, group, target):
    """Group a Pandas DataFrame via a given column and return
    the probability ratio of the target variable for that grouping.

    Args:
        :param df: Pandas DataFrame.
        :param group: Column to group by.
        :param target: Target variable column.

    Returns:
        Probability ratio for the target variable across the group.

    Example:
        df['code_pr'] = get_probability_ratio(df, 'code', 'quantity')

    """

    # Calculate mean of target for group
    prob_ratio = df.groupby(group)[target].mean()
    prob_ratio = pd.DataFrame(prob_ratio)

    # Rename the target as "good"
    prob_ratio = prob_ratio.rename(columns={target: 'good'})

    # Calculate "bad" probability
    prob_ratio['bad'] = 1 - prob_ratio['good']

    # When bad is 0, add a tiny value to avoid division by zero
    prob_ratio['bad'] = np.where(prob_ratio['bad'] == 0, 0.00001, prob_ratio['bad'])

    # Compute probability ratio
    prob_ratio['pr'] = prob_ratio['good'] / prob_ratio['bad']

    return df[group].map(prob_ratio['pr'])


def get_mean_encoding(df, group, target):
    """Group a Pandas DataFrame via a given column and return
    the mean of the target variable for that grouping.

    Args:
        :param df: Pandas DataFrame.
        :param group: Column to group by.
        :param target: Target variable column.

    Returns:
        Mean for the target variable across the group.

    Example:
        df['sector_mean_encoded'] = get_mean_encoding(df, 'sector', 'converted')

    """

    mean_encoded = df.groupby(group)[target].mean()
    return df[group].map(mean_encoded)


def get_frequency_rank(df, column):
    """Return the frequency rank of a categorical variable to
    assign to a new Pandas DataFrame column. This takes the
    value count of each categorical variable and then ranks
    them across the dataframe. Items with equal value counts
    are assigned equal ranking. This is monotonic transformation.

    Args:
        :param df: Pandas DataFrame.
        :param column: Categorical non-numeric column name.

    Returns:
        Frequency rank of the value counts of the column.

    Example:
        df['code_freq_rank'] = get_frequency_rank(df, 'code')

    """
    freq = df[column].value_counts()
    return df[column].map(freq)


def get_conversion_rate(df, total, conversions):
    """Return the conversion rate of column.

    Args:
        :param df: Pandas DataFrame.
        :param total: Column containing the total value.
        :param conversions: Column containing the conversions value.

    Returns:
        Conversion rate of conversions / total

    Example:
        df['cr'] = get_conversion_rate(df, 'sessions', 'orders')

    """

    value = (df[conversions] / df[total])
    return value

