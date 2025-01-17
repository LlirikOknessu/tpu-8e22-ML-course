import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import pickle
import json


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_target_path', '-tp', type=Path, required=True)
    parser.add_argument('--test_features_path', '-fp', type=Path, required=True)
    parser.add_argument('--model_path', '-mp', type=Path, required=True)
    parser.add_argument('--output_path', '-op', type=Path, required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # Загружаем модель
    model: LinearRegression = pickle.load(open(args.model_path, 'rb'))
    X_test = pd.read_csv(args.test_features_path)
    Y_test = pd.read_csv(args.test_target_path)

    Y_pred = model.predict(X_test)
    result = {
        'r2': r2_score(Y_test, Y_pred),
        'mse': mean_squared_error(Y_test, Y_pred),
        'mae': mean_absolute_error(Y_test, Y_pred)
    }

    with open(args.output_path, 'w') as file:
        json.dump(result, file)
