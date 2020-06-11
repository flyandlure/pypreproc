"""
Name: Functions for clustering data
Developer: Matt Clarke
Date: Jan 1, 2020
Description: Allows you to use unsupervised learning within supervised learning models.
"""


import numpy as np
from sklearn.cluster import KMeans


def kmeans_cluster(df, column, cluster_name, n_clusters, fillna_value):
    """Perform K Means clustering on a Pandas DataFrame column and return original DataFrame with a new cluster column.

    Args:
        df: Pandas DataFrame.
        column: Column to cluster.
        cluster_name: Name for new column.
        n_clusters: Number of clusters to create.
        fillna_value: Value to fill NaN values with.

    Returns:
        Original DataFrame with new column containing cluster number.
    """

    # Ensure that inf and NaN values are filled
    df[column].replace([np.inf, -np.inf], np.nan)
    df[column].fillna(value=fillna_value, inplace=True)

    # Cast value to int
    df[column].astype(int)

    # Cluster data
    kmeans = KMeans(n_clusters)
    kmeans.fit(df[[column]])
    df[cluster_name] = kmeans.predict(df[[column]])

    return df

