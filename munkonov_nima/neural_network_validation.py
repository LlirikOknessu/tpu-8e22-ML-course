import pandas as pd
import argparse
from pathlib import Path
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from joblib import load


def parser_args_for_sac():
    parser = argparse.ArgumentParser(description='Paths parser')
    parser.add_argument('--input_dir', '-id',
                        type=str,
                        default='data/prepared/',
                        required=False,
                        help='path to input data directory')
    parser.add_argument('--input_model', '-im',
                        type=str,
                        default='data/models/',
                        required=False,
                        help='path to save prepared data')
    parser.add_argument('--model_name', '-mn',
                        type=str,
                        default='NN',
                        required=False,
                        help='file with dvc stage params')
    parser.add_argument('--logs_path', '-lp',
                        type=str,
                        default='data/logs/',
                        required=False,
                        help='path to logs dir')
    parser.add_argument('--params', '-p',
                        type=str,
                        default='params.yaml',
                        required=False,
                        help='file with dvc stage params')
    return parser.parse_args()


if __name__ == '__main__':
    args = parser_args_for_sac()

    input_dir = Path(args.input_dir)
    input_model = Path(args.input_model)

    X_val_name = input_dir / 'X_val.csv'
    y_val_name = input_dir / 'y_val.csv'

    X_val = pd.read_csv(X_val_name)
    y_val = pd.read_csv(y_val_name)

    reg = load(input_model)
    predicted_values = np.squeeze(reg.predict(X_val))
    r2_score = r2_score(y_val, predicted_values)

    print("NEURAL NETWORK VALIDATION")
    print("Model MAE:    ", mean_absolute_error(y_val, predicted_values))
    print(f"R² Score:      {r2_score:.4f}")
