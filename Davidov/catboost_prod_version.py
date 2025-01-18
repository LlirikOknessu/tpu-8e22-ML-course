import pandas as pd
import argparse
from pathlib import Path
import yaml
from joblib import dump
from catboost import CatBoostRegressor
import random
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def parser_args_for_prod():
    parser = argparse.ArgumentParser(description='CatBoost Production Version Trainer')
    parser.add_argument('--input_dir', '-id', type=str, default='data/prepared/',
                        required=False, help='Path to input data directory')
    parser.add_argument('--output_dir', '-od', type=str, default='data/models/',
                        required=False, help='Path to save the trained model')
    parser.add_argument('--model_name', '-mn', type=str, default='CatBoost_Prod', required=False,
                        help='Name of the output model file')
    parser.add_argument('--params', '-p', type=str, default='params.yaml', required=False,
                        help='Path to the parameters YAML file')
    return parser.parse_args()


if __name__ == '__main__':
    args = parser_args_for_prod()

    with open(args.params, 'r') as f:
        params_all = yaml.safe_load(f)
    # Используем параметры для продакшен-версии
    params = params_all['catboost']["ProdCatBoost"]

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(exist_ok=True, parents=True)
    output_model_joblib_path = output_dir / f"{args.model_name}_prod.joblib"
    output_metrics_path = output_dir / f"{args.model_name}_prod_metrics.json"

    X_train_name = input_dir / 'X_train.csv'
    y_train_name = input_dir / 'y_train.csv'
    X_test_name = input_dir / 'X_test.csv'
    y_test_name = input_dir / 'y_test.csv'

    X_train = pd.read_csv(X_train_name)
    y_train = pd.read_csv(y_train_name).squeeze()  # Преобразование в Series
    X_test = pd.read_csv(X_test_name)
    y_test = pd.read_csv(y_test_name).squeeze()    # Преобразование в Series

    X_full = pd.concat([X_train, X_test], axis=0).reset_index(drop=True)
    y_full = pd.concat([y_train, y_test], axis=0).reset_index(drop=True)

    random.seed(42)

    catboost_model = CatBoostRegressor(**params,loss_function="RMSE",verbose=0)

    catboost_model.fit(X_full, y_full)

    dump(catboost_model, output_model_joblib_path)

    y_pred_full = catboost_model.predict(X_full)

    catboost_model.get_feature_importance()

    feature_importance = catboost_model.get_feature_importance()
    feature_names = X_test.columns
    # Display feature importance
    for name, importance in zip(feature_names, feature_importance):
        print(f"признак: {name}, важность: {importance:.2f}")
