import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import umap
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
import torch
from code.PCF_Autoencoder import train_autoencoder_with_pcf

def impute_time_series_df(data):
    """
    Imputes missing values in a DataFrame using linear interpolation, forward fill, and backward fill.

    Parameters:
    - data (pd.DataFrame): A DataFrame with datetime index and missing values in its columns.

    Returns:
    - pd.DataFrame: A DataFrame where each column has been imputed.
    """
    # Apply linear interpolation first
    data = data.interpolate(method='linear')

    # Apply forward fill
    data = data.ffill()

    # Apply backward fill
    data = data.bfill()

    return data

def scale_dataframe(df, date_column):
    """
    Scales the data using StandardScaler, leaving the date column unchanged.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - date_column (str): The name of the date column in the DataFrame.

    Returns:
    - scaled_df (pd.DataFrame): The scaled DataFrame with the date column unchanged.
    - scaler (StandardScaler): The scaler fitted to the data.
    """
    scaler = StandardScaler()
    data_matrix = df.drop(columns=[date_column]).values
    scaled_data = scaler.fit_transform(data_matrix)
    scaled_df = pd.DataFrame(scaled_data, columns=df.drop(columns=[date_column]).columns)
    scaled_df[date_column] = df[date_column].values
    return scaled_df, scaler

def reduce_dimensions(df, date_column, method='PCA', n_components=2, encoding_dim=10, hidden_dim=32):
    """
    Fits a dimensionality reduction model to the DataFrame and returns the transformed data along with the model.
    """
    date_data = df[date_column]
    data_matrix = df.drop(columns=[date_column]).values

    if method == 'PCA':
        reducer = PCA(n_components=n_components)
        result = reducer.fit_transform(data_matrix)
    elif method == 'UMAP':
        reducer = umap.UMAP(n_components=n_components, random_state=42, n_jobs=1)
        result = reducer.fit_transform(data_matrix)
    elif method == 'PCF-AE':
        encoder, scaler = train_autoencoder_with_pcf(df, date_column, encoding_dim=encoding_dim, hidden_dim=hidden_dim)
        scaled_data = scaler.transform(data_matrix)
        scaled_data = torch.tensor(scaled_data.reshape(scaled_data.shape[0], 1, scaled_data.shape[1]), dtype=torch.float32)
        encoded, _ = encoder(scaled_data)  # Get only the encoded part
        result = encoded.detach().numpy()
        reducer = (encoder, scaler)  # Returning both encoder and scaler as the reducer
    else:
        raise ValueError("Unsupported dimensionality reduction method specified.")

    result_df = pd.DataFrame(result, columns=[f'{method}{i+1}' for i in range(result.shape[1])])
    result_df[date_column] = date_data.reset_index(drop=True)
    
    return result_df, reducer

def apply_dimension_reduction(df, date_column, reducer, method, n_components):
    """
    Applies a fitted dimensionality reduction model to the DataFrame.
    """
    date_data = df[date_column]
    data_matrix = df.drop(columns=[date_column]).values

    if method == 'PCF-AE':
        encoder, scaler = reducer
        scaled_data = scaler.transform(data_matrix)
        scaled_data = torch.tensor(scaled_data.reshape(scaled_data.shape[0], 1, scaled_data.shape[1]), dtype=torch.float32)
        encoded, _ = encoder(scaled_data)  # Get only the encoded part
        result = encoded.detach().numpy()
    else:
        result = reducer.transform(data_matrix)

    result_df = pd.DataFrame(result, columns=[f'{method}{i+1}' for i in range(n_components)])
    result_df[date_column] = date_data.reset_index(drop=True)
    
    return result_df

def perform_time_series_predictions(df, outcome_var, models_to_run, date_column, test_size=0.2):
    """
    Predict future points in a time series dataset using various regression models.
    """
    # Prepare the data
    df['y_t+1'] = df[outcome_var].shift(-1)
    df.dropna(inplace=True)

    results = {}
    split_point = int(len(df) * (1 - test_size))
    X_train = df.iloc[:split_point].drop(columns=['y_t+1', date_column])
    y_train = df.iloc[:split_point]['y_t+1']
    X_test = df.iloc[split_point:].drop(columns=['y_t+1', date_column])
    y_test = df.iloc[split_point:]['y_t+1']

    # Initialize models
    models = {
        'OLS': LinearRegression(),
        'RF': RandomForestRegressor(n_estimators=100, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=100, random_state=42),
        'NN': Sequential([
            Input(shape=(X_train.shape[1],)), 
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(1)
        ])
    }

    # Fit and evaluate models
    for model_name in models_to_run:
        model = models[model_name]
        if model_name == 'NN':  # Specific handling for neural network
            model.compile(optimizer=Adam(), loss='mean_squared_error')
            model.fit(X_train, y_train, epochs=50, batch_size=10, verbose=0)
            y_pred = model.predict(X_test).flatten()
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        r_squared = r2_score(y_test, y_pred)
        results[model_name] = {'Mean Squared Error': mse, 'r_squared': r_squared}
    
    return results

def evaluate_models_with_dimensionality_reduction(df, outcome_var, date_column, start_date, rolling_window):
    # methods = ['PCA', 'UMAP', 'PCF-AE']
    methods = ['PCF-AE']
    models_to_run = ['OLS', 'RF', 'XGBoost']
    models_to_run = ['OLS']
    components_to_run = [1, 3, 5, 20, 50]

    # Ensure the date_column is in datetime format
    df[date_column] = pd.to_datetime(df[date_column])
    start_date = pd.to_datetime(start_date)

    # List to hold all results
    results_list = []
    
    date_range = pd.date_range(start=start_date, end=df[date_column].max(), freq=f'{rolling_window}MS')

    for start in date_range:
        print(f"Processing start date: {start}, outcome variable: {outcome_var}")
        train_df = df[(df[date_column] < start)]
        test_df = df[(df[date_column] >= start) & (df[date_column] < start + pd.DateOffset(months=rolling_window))]
        
        if len(test_df) == 0:
            break
        
        for method in methods:
            for n_components in components_to_run:
                print(f"  Method: {method}, n_components: {n_components}")
                # Fit dimension reduction on training data and transform both training and test data
                reduced_train_df, reducer = reduce_dimensions(train_df, date_column, method=method, n_components=n_components)
                reduced_test_df = apply_dimension_reduction(test_df, date_column, reducer, method, n_components)

                # Combine training and test data back for time series prediction as needed
                reduced_df = pd.concat([reduced_train_df, reduced_test_df])
                reduced_df[outcome_var] = df[outcome_var]

                # Perform predictions and collect results
                model_results = perform_time_series_predictions(reduced_df, outcome_var, models_to_run, date_column, test_size=0.2)

                # Store the R-squared results for each model using the current method and number of components
                for model in models_to_run:
                    results_list.append({
                        'Outcome': outcome_var,
                        'Reduction Model': method,
                        'Prediction Model': model,
                        'N-Component': n_components,
                        'R2': model_results[model]['r_squared']
                    })

    # Create a DataFrame from the list of results
    results_df = pd.DataFrame(results_list)
    return results_df

# Read training data
df = pd.read_csv('Dataset/M3/M3_monthly.csv')
low_na = [column for column in df.columns if df[column].isnull().mean() < 0.1 and column != 'timestamp']
print(low_na)
# Impute dataframe
df_imp = impute_time_series_df(df)

# Scale dataframe
df_imp, scaler = scale_dataframe(df_imp, 'timestamp')

all_results = []
for y in low_na[:2]:
    # Evaluate models for each outcome variable
    print(f"Evaluating models for outcome variable: {y}")
    results_table = evaluate_models_with_dimensionality_reduction(df_imp, y, 'timestamp', '1990-01-01', 12)
    
    # Append the results to the list
    all_results.append(results_table)

# Concatenate all result DataFrames into a single DataFrame
final_results = pd.concat(all_results, ignore_index=True)
print("Saving final results to CSV")
final_results.to_csv('results/baseline2.csv')
print("Processing complete")
