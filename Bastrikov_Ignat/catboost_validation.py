import pandas as pd
import argparse
from pathlib import Path
import yaml
from joblib import load
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.base import RegressorMixin


def parser_args_for_validation():
    parser = argparse.ArgumentParser(description='CatBoost Validation Script')
    parser.add_argument('--input_dir', '-id', type=str, default='data/prepared/',
                        required=False, help='Path to input data directory')
    parser.add_argument('--model_path', '-mp', type=str, required=True,
                        help='Path to the trained CatBoost model (.joblib)')
    parser.add_argument('--baseline_model', '-bm', type=str, default='data/models/LinearRegression_prod.joblib',
                        required=False, help='Path to the baseline model (e.g., linear regression)')
    parser.add_argument('--model_name', '-mn', type=str, default='CatBoost',
                        required=False, help='Name of the model')
    parser.add_argument('--params', '-p', type=str, default='params.yaml',
                        required=False, help='File with parameters')
    return parser.parse_args()


if __name__ == '__main__':
    args = parser_args_for_validation()

    with open(args.params, 'r') as f:
        params_all = yaml.safe_load(f)

    input_dir = Path(args.input_dir)
    model_path = Path(args.model_path)
    baseline_model_path = Path(args.baseline_model)

    catboost_model = load(model_path)
    baseline_model = load(baseline_model_path)

    X_test_path = input_dir / 'X_test.csv'
    y_test_path = input_dir / 'y_test.csv'
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path).squeeze()

    y_pred_baseline = baseline_model.predict(X_test)
    y_pred_catboost = catboost_model.predict(X_test)

    mae_baseline = mean_absolute_error(y_test, y_pred_baseline)
    mse_baseline = mean_squared_error(y_test, y_pred_baseline)
    r2_baseline = r2_score(y_test, y_pred_baseline)

    mae_catboost = mean_absolute_error(y_test, y_pred_catboost)
    mse_catboost = mean_squared_error(y_test, y_pred_catboost)
    r2_catboost = r2_score(y_test, y_pred_catboost)

    print(f"Валидация модели CatBoost ({model_path.stem})")
    print("-------------------------------")
    print("Базовая Модель:")
    print(f"  MAE: {mae_baseline:.4f}")
    print(f"  MSE: {mse_baseline:.4f}")
    print(f"  R2: {r2_baseline:.4f}\n")

    print("Модель CatBoost:")
    print(f"  MAE: {mae_catboost:.4f}")
    print(f"  MSE: {mse_catboost:.4f}")
    print(f"  R2: {r2_catboost:.4f}")
