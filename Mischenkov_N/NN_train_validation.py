import argparse
import yaml
import pandas as pd
import numpy as np
import shutil
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import datetime

def parser_args_for_sac():
    parser = argparse.ArgumentParser(description='Paths parser')
    parser.add_argument('--input_dir', '-id', type=str, default='./data/prepared/',
                        required=False, help='path to input data directory')
    parser.add_argument('--output_dir', '-od', type=str, default='./data/models/',
                        required=False, help='path to save prepared data')
    parser.add_argument('--model_dir', '-md', type=str, default='./data/models/',
                        required=False, help='path to the directory containing the model')
    parser.add_argument('--params', '-p', type=str, default='params.yaml', required=False,
                        help='file with dvc stage params')
    return parser.parse_args()

def load_data(input_dir):
    X_val = pd.read_csv(Path(input_dir) / 'X_val.csv')
    y_val = pd.read_csv(Path(input_dir) / 'y_val.csv').squeeze()
    return X_val, y_val


if __name__ == '__main__':
    args = parser_args_for_sac()
    with open(args.params, 'r') as f:
        params_all = yaml.safe_load(f)
    params = params_all['Neural_network']

    input_dir = Path(args.input_dir)
    model_dir = Path(args.model_dir)
    logs_path = Path('./data/logs')

    # Проверка наличия директорий
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory {input_dir} does not exist.")
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory {model_dir} does not exist.")

    # Загрузка данных
    X_val, y_val = load_data(input_dir)

    # Загрузка модели
    model_path = model_dir / 'nn_model.h5'
    if not model_path.exists():
        raise FileNotFoundError(f"Model file {model_path} does not exist.")

    model = load_model(model_path, custom_objects={'mse': MeanSquaredError()})

    y_pred = model.predict(X_val.to_numpy()).flatten()

    mae = mean_absolute_error(y_val, y_pred)
    mse = mean_squared_error(y_val, y_pred)
    rmse = np.sqrt(mse)

    metrics = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse
    }
    print("=== Metrics of NN ===")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv('data/prepared/metrics_validation.csv', index=False)
    print(f"Метрики сохранены в data/prepared/metrics_validation.csv")