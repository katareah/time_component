import pandas as pd

df = pd.read_csv('Dataset/M3/M3_monthly_TSTS_raw.csv')

# Remove rows where 'timestamp' is NaN
df = df.dropna(subset=['timestamp'])

# Set 'timestamp' as the index
df.set_index('timestamp', inplace=True)

# Check if DataFrame is empty after dropping NaT (as in this example, all are NaT)
if df.empty:
    print("DataFrame is empty after removing rows with NaN timestamps.")
else:
    # Concatenate 'series_id' and 'category' to create new column names
    df['new_column'] = df['series_id'] + '_' + df['category']

    # Pivot the DataFrame to widen it
    df_pivot = df.pivot(columns='new_column', values='value')

    # Display the reshaped DataFrame
    print("Reshaped DataFrame:")
    print(df_pivot)

# Calculate the number of NA values per row
na_counts = df_pivot.isna().sum(axis=1)

# Calculate the total number of columns
total_columns = df_pivot.shape[1]

# Calculate NA percentage for each row
df_pivot['na_percentage'] = (na_counts / total_columns) * 100
# Filter out rows where the NA percentage is 50% or more
df_filtered = df_pivot[df_pivot['na_percentage'] < 50]
df_filtered = df_filtered.loc[:, df_filtered.iloc[-1].notna()]
df_filtered = df_filtered.drop('na_percentage', axis = 1)
df_filtered.to_csv('Dataset/M3/M3_monthly.csv')