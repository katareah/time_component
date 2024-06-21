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
from src.PCFGAN.PCFGAN import RPCFGANTrainer
from src.networks.discriminators import LSTMDiscriminator
from src.networks.generators import LSTMGenerator
from src.evaluations.test_metrics import get_standard_test_metrics
from src.utils import loader_to_tensor
import ml_collections

# Define the configuration
def get_config():
    config = ml_collections.ConfigDict()
    config.dataset = 'Stock'  # Choose from 'ROUGH', 'Air_Quality', 'Stock', 'OU', 'EEG'
    config.n_lags = 10
    config.batch_size = 64
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config.G_input_dim = 100
    config.G_hidden_dim = 64
    config.G_num_layers = 2
    config.D_hidden_dim = 64
    config.D_num_layers = 2
    config.D_out_dim = 64
    config.steps = 10000
    config.lr_G = 0.0002
    config.lr_D = 0.0002
    config.lr_M = 0.0002
    config.gamma = 0.99
    config.D_steps_per_G_step = 1
    config.M_num_samples = 5
    config.M_hidden_dim = 64
    config.init_range = 1.0
    config.Lambda1 = 0.1
    config.Lambda2 = 0.1
    config.swa_step_start = 5000
    config.exp_dir = './experiments'
    config.grad_clip = 1.0
    config.BM = False
    config.noise_scale = 0.3
    config.add_time = True
    config.input_dim = 10
    config.generator = 'LSTM'
    config.discriminator = 'LSTM'
    config.dataset = "Stock"  # Change this to the desired dataset
    return config

def impute_time_series_df(data):
    data = data.interpolate(method='linear')
    data = data.ffill()
    data = data.bfill()
    return data

def scale_dataframe(df, date_column):
    scaler = StandardScaler()
    data_matrix = df.drop(columns=[date_column]).values
    scaled_data = scaler.fit_transform(data_matrix)
    scaled_df = pd.DataFrame(scaled_data, columns=df.drop(columns=[date_column]).columns)
    scaled_df[date_column] = df[date_column].values
    return scaled_df, scaler

def get_dataset(config, num_workers=1, data_root="./data"):
    dataset = {
        "ROUGH": Rough_S,
        "Air_Quality": Beijing_air_quality,
        "Stock": Stock,
        "OU": OU,
        "EEG": EEG,
    }[config.dataset]
    training_set = dataset(partition="train", n_lags=config.n_lags)
    test_set = dataset(partition="test", n_lags=config.n_lags)

    training_loader = DataLoader(
        training_set, batch_size=config.batch_size, shuffle=True, num_workers=num_workers, drop_last=True
    )
    test_loader = DataLoader(
        test_set, batch_size=config.batch_size, shuffle=True, num_workers=num_workers, drop_last=True
    )

    config.input_dim = training_loader.dataset[0][0].shape[-1]
    n_lags = next(iter(test_loader))[0].shape[1]
    config.update({"n_lags": n_lags}, allow_val_change=True)
    return training_loader, test_loader

def get_trainer(config, train_dl, test_dl=None):
    x_real_train = loader_to_tensor(train_dl).to(config.device)
    if test_dl:
        x_real_test = loader_to_tensor(test_dl).to(config.device)
    else:
        x_real_test = None

    D_out_dim = config.D_out_dim
    return_seq = True
    activation = nn.Tanh() if config.dataset != "OU" else nn.Identity()

    generator = LSTMGenerator(
        input_dim=config.G_input_dim,
        hidden_dim=config.G_hidden_dim,
        output_dim=config.input_dim,
        n_layers=config.G_num_layers,
        noise_scale=config.noise_scale,
        BM=config.BM,
        activation=activation,
    )
    discriminator = LSTMDiscriminator(
        input_dim=config.input_dim,
        hidden_dim=config.D_hidden_dim,
        out_dim=D_out_dim,
        n_layers=config.D_num_layers,
        return_seq=return_seq,
    )

    test_metrics_train = get_standard_test_metrics(x_real_train)
    test_metrics_test = get_standard_test_metrics(x_real_test) if x_real_test else None

    trainer = RPCFGANTrainer(
        G=generator,
        D=discriminator,
        test_metrics_train=test_metrics_train,
        test_metrics_test=test_metrics_test,
        train_dl=train_dl,
        batch_size=config.batch_size,
        n_gradient_steps=config.steps,
        config=config,
    )

    torch.backends.cudnn.benchmark = True
    return trainer, generator

def reduce_dimensions(df, date_column, method='PCA', n_components=2, config=None):
    date_data = df[date_column]
    data_matrix = df.drop(columns=[date_column]).values

    if method == 'PCA':
        reducer = PCA(n_components=n_components)
        result = reducer.fit_transform(data_matrix)
    elif method == 'UMAP':
        reducer = umap.UMAP(n_components=n_components, random_state=42, n_jobs=1)
        result = reducer.fit_transform(data_matrix)
    elif method == 'RPCFGAN':
        if config is None:
            raise ValueError("Config must be provided for RPCFGAN.")
        train_dl = DataLoader(TensorDataset(torch.tensor(data_matrix, dtype=torch.float32)), batch_size=config.batch_size, shuffle=True)
        trainer, generator = get_trainer(config, train_dl)
        device = config.device
        trainer.fit(device)

        encoder = generator
        scaler = StandardScaler().fit(data_matrix)
        scaled_data = scaler.transform(data_matrix)
        scaled_data = torch.tensor(scaled_data.reshape(scaled_data.shape[0], 1, scaled_data.shape[1]), dtype=torch.float32)
        encoded, _ = encoder(scaled_data)
        result = encoded.detach().numpy()
        reducer = (encoder, scaler)
    else:
        raise ValueError("Unsupported dimensionality reduction method specified.")

    result_df = pd.DataFrame(result, columns=[f'{method}{i+1}' for i in range(result.shape[1])])
    result_df[date_column] = date_data.reset_index(drop=True)

    return result_df, reducer

def apply_dimension_reduction(df, date_column, reducer, method, n_components):
    date_data = df[date_column]
    data_matrix = df.drop(columns=[date_column]).values

    if method == 'RPCFGAN':
        encoder, scaler = reducer
        scaled_data = scaler.transform(data_matrix)
        scaled_data = torch.tensor(scaled_data.reshape(scaled_data.shape[0], 1, scaled_data.shape[1]), dtype=torch.float32)
        encoded, _ = encoder(scaled_data)
        result = encoded.detach().numpy()
    else:
        result = reducer.transform(data_matrix)

    result_df = pd.DataFrame(result, columns=[f'{method}{i+1}' for i in range(n_components)])
    result_df[date_column] = date_data.reset_index(drop=True)

    return result_df

def perform_time_series_predictions(df, outcome_var, models_to_run, date_column, test_size=0.2):
    df['y_t+1'] = df[outcome_var].shift(-1)
    df.dropna(inplace=True)

    results = {}
    split_point = int(len(df) * (1 - test_size))
    X_train = df.iloc[:split_point].drop(columns=['y_t+1', date_column])
    y_train = df.iloc[:split_point]['y_t+1']
    X_test = df.iloc[split_point:].drop(columns=['y_t+1', date_column])
    y_test = df.iloc[split_point:]['y_t+1']

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

    for model_name in models_to_run:
        model = models[model_name]
        if model_name == 'NN':
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

def evaluate_models_with_dimensionality_reduction(df, outcome_var, date_column, start_date, rolling_window, config=None):
    methods = ['PCA', 'UMAP', 'RPCFGAN']
    models_to_run = ['OLS', 'RF', 'XGBoost']
    components_to_run = [1, 3, 5, 20, 50]

    df[date_column] = pd.to_datetime(df[date_column])
    start_date = pd.to_datetime(start_date)

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
                reduced_train_df, reducer = reduce_dimensions(train_df, date_column, method=method, n_components=n_components, config=config)
                reduced_test_df = apply_dimension_reduction(test_df, date_column, reducer, method, n_components)

                reduced_df = pd.concat([reduced_train_df, reduced_test_df])
                reduced_df[outcome_var] = df[outcome_var]

                model_results = perform_time_series_predictions(reduced_df, outcome_var, models_to_run, date_column, test_size=0.2)

                for model in models_to_run:
                    results_list.append({
                        'Outcome': outcome_var,
                        'Reduction Model': method,
                        'Prediction Model': model,
                        'N-Component': n_components,
                        'R2': model_results[model]['r_squared']
                    })

    results_df = pd.DataFrame(results_list)
    return results_df

# Main script
if __name__ == "__main__":
    # Define the configuration
    config = get_config()

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
        print(f"Evaluating models for outcome variable: {y}")
        results_table = evaluate_models_with_dimensionality_reduction(df_imp, y, 'timestamp', '1990-01-01', 12, config=config)
        all_results.append(results_table)

    final_results = pd.concat(all_results, ignore_index=True)
    print("Saving final results to CSV")
    final_results.to_csv('results/baseline2.csv')
    print("Processing complete")
