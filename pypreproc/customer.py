"""
Name: Functions for customer data
Developer: Matt Clarke
Date: Jan 1, 2020
Description: Specific functions for generating customer data including RFM scores.
"""

from lifetimes.utils import summary_data_from_transaction_data


def rfm_model(df, customer_column, date_column, monetary_column):
    """Return an RFM score for each customer using the Lifetimes RFM model.
    This score is calculated across the whole DataFrame, so if you have a
    customer with numerous orders, it will calculate one value and apply
    it across all orders and won't calculate the figure historically.

    Args:
        :param df: Pandas DataFrame
        :param monetary_column: Column containing monetary value of order
        :param date_column: Column containing date
        :param customer_column: Column containing customer

    Returns:
        New DataFrame containing RFM data by customer.
        T is equal to days since first order and end of period.
        Customers with 1 order will be assigned 0 for RFM scores.
    """

    # Ensure that inf and NaN values are filled
    rfm_df = summary_data_from_transaction_data(df,
                                                customer_column,
                                                date_column,
                                                monetary_value_col=monetary_column)
    return rfm_df

