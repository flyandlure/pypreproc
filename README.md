# PyPreProc
PyPreProc is a Python package that you can use for preprocessing and feature engineering during machine learning development projects. It uses Pandas and makes it quicker and easier to correct, convert, cluster and create data. 

## Setup
PyPreProc can be installed via PyPi using `pip3 install pypreproc`. (Some versions of `wheel` may prevent installation. If this happens, run `pip3 install wheel --upgrade` to use a later version and then install with `pip3 install pypreproc`.)

## Examples
To use PyPreProc simply load your data into a Pandas DataFrame.
To use PyPreProc simply load your data into a Pandas DataFrame.

```python
import pandas as pd
from pypreproc import correct, convert, create, cluster, customer
df = pd.read_csv('data.csv')
```

### Correcting data
The `cols_to_strip_characters()` function takes a list of columns and a character or string and strips this from the column data and returns the modified DataFrame. 

```python
strip_cols = ['name', 'address']
df = correct.cols_to_strip_characters(df, strip_cols, '$')
```

The `cols_to_drop()` function drops a list of columns and returns the modified DataFrame.

```python
drop_cols = ['email', 'shoe_size']
df = correct.cols_to_drop(df, drop_cols)
```

The `col_names_to_lower()` function converts all Pandas column names to lowercase. 

```python
df = correct.col_names_to_lower(df)
```

The `cols_to_rename()` function lets you use a Python dictionary to rename specific column names in your Pandas DataFrame. 

```python
rename_cols = {'old_name1': 'new_name1', 'old_name2': 'new_name2'}
df = correct.cols_to_rename(df, rename_cols)
```

### Converting data

The `cols_to_slugify()` function "slugifies" data into a continuous string, stripping special characters and replacing spaces with underscores. It's very useful for use with one-hot encoding as the column values become column names that are easier to reference. 

```python
slugify_cols = ['country', 'manufacturer']
df = convert.cols_to_slugify(df, slugify_cols)
```

The `cols_to_float()` function converts column values to `float`. 

```python
float_cols = ['price', 'weight']
df = convert.cols_to_float(df, float_cols)
```

The `cols_to_int()` function converts column values to `int`. 

```python
int_cols = ['age', 'children']
df = convert.cols_to_float(df, int_cols)
```

The `cols_to_datetime()` function converts column values to `datetime`. 

```python
date_cols = ['date']
df = convert.cols_to_float(df, date_cols)
```

The `cols_to_negative()` function converts column values to a negative value. 

```python
neg_cols = ['score']
df = convert.cols_to_negative(df, neg_cols)
```

### Clustering data
The `kmeans_cluster()` function makes it easy to use unsupervised learning algorithms in your supervised machine learning model, which can often yield great improvements. 

To use this function you pass the DataFrame, the column you wish to cluster, provide a name for the new column of cluster data, define the number of cluster to create and provide a value to add if there is a `NaN` returned. 

```python
# Definitions added for readability
column = 'offspring'
cluster_name = 'fecundity'
n_clusters = 5
fillna_value = 0

df = cluster.kmeans_cluster(df, 
column, 
cluster_name, 
n_clusters, 
fillna_value)
```

### Customer data
The `rfm_model()` function uses the excellent Lifetimes package to perform simple RFM modelling. This examines the Recency, Frequency and Monetary values of customers and returns data that identify the value of the customer and their propensity to purchase again. 

```python
# Definitions added for readability
customer_column = 'customer_id'
date_column = 'order_date'
monetary_column = 'order_value'

df = customer.rfm_model(df, customer_column, date_column, monetary_column)
```

### Creating data
The `create` module includes a wide range of functions for creating new features from existing data in your Pandas DataFrame. These are of use when investigating which features might have a correlation with your model's target and could improve model performance.

The `cols_to_log()` function takes a list of columns and provides their log values in new columns prefixed with `log_` so you can compare them against the non-transformed data. This method is best for data where the column values do not include zeros. 

```python
log_cols = ['engine_size', 'output']
df = create.cols_to_log(df, log_cols)
```

The `cols_to_log1p()` function works like `cols_to_log()` but adds 1 allowing it to work on columns that contain zero values. 

```python
log_cols = ['fins', 'gill_rakers']
df = create.cols_to_log1p(df, log_cols)
```

The `cols_to_log_max_root()` function convert column data to log values using the maximum value as the log max and returns new columns of prefixed data. For use with data where the column values include zeros.

```python
log_cols = ['fins', 'gill_rakers']
df = create.cols_to_log_max_root(df, log_cols)
``` 

The `cols_to_tanh()` function takes a list of columns and returns their hyperbolic tangent in a new column. 

```python
cols = ['fins', 'gill_rakers']
df = create.cols_to_tanh(df, cols)
``` 

The `cols_to_sigmoid()` function takes a list of columns and creates data points to values between 0 and 1 using a sigmoid function and return new columns of prefixed data. 

```python
cols = ['fins', 'gill_rakers']
df = create.cols_to_sigmoid(df, cols)
``` 

The `cols_to_cube_root()` function takes a list of columns and returns their cube root so all values are between 0 and 1. 

```python
cols = ['fins', 'gill_rakers']
df = create.cols_to_cube_root(df, cols)
``` 

The `cols_to_cube_root_normalize()` function takes a list of columns and returns their normalised cube root so all values are between 0 and 1. 

```python
cols = ['fins', 'gill_rakers']
df = create.cols_to_cube_root_normalize(df, cols)
``` 

The `cols_to_percentile()` function converts data points to their percentile linearized value and return new columns of prefixed data.

```python
cols = ['fins', 'gill_rakers']
df = create.cols_to_percentile(df, cols)
``` 

The `cols_to_normalize()` function normalizes data points to values between 0 and 1 and return new columns of prefixed data.

```python
cols = ['fins', 'gill_rakers']
df = create.cols_to_normalize(df, cols)
``` 

The `cols_to_log1p_normalize()` function log+1 normalizes data points to values between 0 and 1 and return new columns of prefixed data. It's best for use with columns where data contain zeros.

```python
cols = ['fins', 'gill_rakers']
df = create.cols_to_log1p_normalize(df, cols)
``` 

The `cols_to_one_hot()` function one-hot encodes column values and creates new columns containing the one-hot encoded data. For example, if you have a column containing two values (fish or bird) it will return a 1 or 0 for bird and a 1 or 0 for fish. 

It's designed for use with low cardinality data (in which there are only a small number of values within the column).

```python
cols = ['class', 'genus']
df = create.cols_to_one_hot(df, cols)
``` 

The `cols_to_reduce_uniques()` function takes a list of columns and reduces the number of unique values by assigning those below a given threshold as "others".

```python
cols = {'col1' : 1000, 'col2' : 3000}
df = create.cols_to_reduce_uniques(df, cols)
``` 

#### Grouped data

The `cols_to_count_uniques()` counts the number of unique column values when grouping by another column and return new columns in original DataFrame. For example, if grouping by the column `region` and examining data in the `cars` and `children` columns the function would return new columns called `unique_cars_by_region`.  

```python
df = create.cols_to_count_uniques(df, 'region', ['cars', 'children'])
```

The `cols_to_count()` function counts the number of column values when grouping by another column and return new columns in original DataFrame.

```python
df = create.cols_to_count(df, 'region', 'cars')
```

The `cols_to_sum()` function sums the value of column values when grouping by another column and return new columns in original DataFrame.

```python
df = create.cols_to_sum(df, 'region', 'cars')
```

The `get_rolling_average()` function returns the rolling average for a column based on a grouping over X previous periods. For example, the rolling average order value for a customer over their past three visits. 

```python
df['ravg'] = create.get_rolling_average(df, 'group_col', 'avg_col', 5, 'sort_col')
```

The `get_grouped_metric()` function performs a specified mathematical operation on a metric column using a group column. For example, summing all of the sessions by a user ID within the DataFrame.

```python
df['total_sessions'] = create.get_grouped_metric(df, 'id', 'sessions', 'count')
```



### Dates

The `get_days_since_date()` function returns a new column containing the date difference in days between two dates. For example, the number of days since a last dose.

```python
df = create.get_days_since_date(df, 'date_before', 'date_after', 'days_since_last_dose')
```

The `get_dates()` function takes a single date column and returns a load of new columns including: `day`, `month`, `year`, `year_month`, `week_number`, `day_number`, `day_name`, `month_name`, `mysql_date`, `quarter`, which are often more useful in modeling than a very granular date. 

```python
df = create.get_dates(df, 'visit_date')
```

The `date_add()` and `date_subtract` functions add or subtract a given number of days from a date and return a new date in the desired format. The date provided can be a column value from a DataFrame row or the current date. 

```python
date_minus_7 = create.date_subtract(datetime.today(), 7, '%Y-%m-%d')
date_plus_7 = create.date_subtract(datetime.today(), 7, '%Y-%m-%d')
```

### Other features

The `get_grouped_stats()` function groups data by a column and returns summary statistics for a list of columns and add prefixed data to new columns. These include: mean, median, std, max and min. 

```python
df = create.get_grouped_stats(df, 'species', ['dorsal_fin_rays', 'lateral_line_scales'])
```

The `get_feature_interactions()` function combines multiple features to create new features based on 2, 3 or 4 unique combinations. 

```python
df = create.get_feature_interactions(df, ['fins','scales','gill_rakers'], 3)
```

The `get_binned_data()` function performs a simple binning operation on a column and return a column of binned data in DataFrame.

```python
# Definitions added for readability
column = 'orders'
name = 'orders_bin'
bins = 5

df = create.get_binned_data(df, column, name, bins)
```

The `sum_columns()` function returns a single value based on the sum of a column.

```python
value = create.sum_columns(df, 'revenue')
```

The `get_diff()` function returns the difference between two column values. 

```python
value = create.get_diff(df, 'order_value', 'aov')
```

The `get_previous_cumulative_sum()` function gets the previous cumulative sum of a column based on a group. For example, the running total of orders placed by a given customer at time X. It does not include the current value.  

```python
previous_orders = create.get_previous_cumulative_sum(df, 'group_column', 'sum_column', 'sort_column')
```

The `get_cumulative_sum()` function returns the cumulative sum of a grouped column and includes the current value. 

```python
total_orders = create.get_cumulative_sum(df, 'group_column', 'sum_column', 'sort_column')
```

The `get_previous_cumulative_count()` function counts cumulative column values based on a grouping, not including the current value. 

```python
previous_orders = create.get_previous_cumulative_count(df, 'group_column', 'count_column', 'sort_column')
```

The `get_previous_value()` function groups by a column and return the previous value of another column and assign value to a new column. For example, the previous value of a customer's order.

```python
df = create.get_previous_value(df, 'customer_id', 'order_value')
```

The `get_probability_ratio()` groups a Pandas DataFrame via a given column and returns the probability ratio of the target variable for that grouping. It's a useful way of using target data to improve model performance, with less likelihood of introducing data leakage.  

```python
df = create.get_probability_ratio(df, 'group_column', 'target_column')
```

The `get_mean_encoding()` function groups a Pandas DataFrame via a given column and returns the mean of the target variable for that grouping. For example, if your model's target variable is "revenue" what is the mean revenue for people by "country"?

```python
df = create.get_mean_encoding(df, 'group_column', 'target_column')
```

The `get_frequency_rank()` function return the frequency rank of a categorical variable to assign to a new Pandas DataFrame column. This takes the value count of each categorical variable and then ranks them across the DataFrame. Items with equal value counts are assigned equal ranking. This is monotonic transformation.

```python
df['freq_rank_species'] = create.get_frequency_rank(df, 'lateral_line_pores')
```

The `get_conversion_rate()` function returns the conversion rate for a column. 

```python
df['cr'] = create.get_conversion_rate(df, 'sessions', 'orders')
```

The `count_subzero()` function returns a count of values that are less than one. By filling NaN values with -1, and then counting these, you can create a new feature. 

```python
df['total_subzero'] = create.count_subzero(df)
```

Similarly, `count_zero()` does the same for zero values. 

```python
df['total_zero'] = create.count_zero(df)
```

The `count_missing()` function does this for NaN values. 

```python
df['total_nan'] = create.count_missing(df)
```

## Helpers
The `select()` helper function provides a very quick and easy way to filter a Pandas DataFrame. It takes five values: `df` (the DataFrame), `column_name` (the name of the column you want to search), `operator` (the search operator you want to use [endswith, startswith, contains, isin, is, gt, gte, lt, lte]), and an optional `exclude` parameter (`True` or `False`) which defines whether the search includes or excludes the data. The gt, lt, lte, gte operators can only be used on numeric or date fields,

```python
ends = helper.select(df, 'genus', 'endswith', where='chromis', exclude=False)
sw = helper.select(df, 'genus', 'startswith', where='Haplo', exclude=False)
contains = helper.select(df, 'genus', 'contains', where='cich', exclude=False)
isin = helper.select(df, 'genus', 'isin', where=['cich','theraps'], exclude=False)
_is = helper.select(df, 'genus', 'is', where='Astronotus', exclude=False)
date_gte = helper.select(df, 'date', 'date_gte', '2020-06-05')
date_gt = helper.select(df, 'date', 'date_gt', '2020-06-05')
date_lte = helper.select(df, 'date', 'date_lte', '2020-06-05')
date_lt = helper.select(df, 'date', 'date_lt', '2020-06-05')
``` 

The `get_unique_rows()` function de-dupes rows in Pandas DataFrame and returns a new DataFrame containing only the unique rows. This simple function keeps the last value. 

```python
df = helper.get_unique_rows(df, ['col1', 'col2'], 'sort_column')
```

The `get_low_var_cols()` function examines all of the numeric columns in a Pandas DataFrame and returns those which have variances lower than a defined threshold. These low variance columns are good candidates for removal from your model, since they will contribute little.

```python
low_var_cols = helper.get_low_var_cols(df, 0.01)
``` 

To get a list of numeric columns in a Pandas DataFrame you can use the `get_numeric_cols()` function.

```python
numeric_cols = helper.get_numeric_cols(df)
```

Similarly, `get_non_numeric_cols()` returns a list of non-numeric column names.

```python
non_numeric_cols = helper.get_non_numeric_cols(df)
```
