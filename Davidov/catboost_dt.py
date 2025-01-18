import pandas as pd
import argparse
from pathlib import Path
import yaml
import numpy as np
from joblib import dump, load
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.base import RegressorMixin
from catboost import CatBoostRegressor
import random


def parser_args_for_sac():
    parser = argparse.ArgumentParser(description='Paths parser')
    parser.add_argument('--input_dir', '-id', type=str, default='data/prepared/',
                        required=False, help='path to input data directory')
    parser.add_argument('--output_dir', '-od', type=str, default='data/models/',
                        required=False, help='path to save prepared data')
    parser.add_argument('--baseline_model', '-bm', type=str, default='data/models/LinearRegression_prod.joblib',
                        required=False, help='path to linear regression prod version')
    parser.add_argument('--model_name', '-mn', type=str, default='LR', required=False,
                        help='file with dvc stage params')
    parser.add_argument('--params', '-p', type=str, default='params.yaml', required=False,
                        help='file with dvc stage params')
    return parser.parse_args()


if __name__ == '__main__':
    args = parser_args_for_sac()

    with open(args.params, 'r') as f:
        params_all = yaml.safe_load(f)
    params = params_all['catboost']["TrainCatBoost"]

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    baseline_model_path = Path(args.baseline_model)

    output_dir.mkdir(exist_ok=True, parents=True)
    output_model_joblib_path = output_dir / f"{args.model_name}.joblib"

    X_train_path = input_dir / 'X_train.csv'
    y_train_path = input_dir / 'y_train.csv'
    X_test_path = input_dir / 'X_test.csv'
    y_test_path = input_dir / 'y_test.csv'

    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path).squeeze()  # Преобразование в Series
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path).squeeze()  # Преобразование в Series

    random.seed(42)

    # отвечает за подбор лучшей модели, в  params должны быть массивы с
    grid_search = GridSearchCV(estimator=CatBoostRegressor(loss_function="RMSE",verbose=0),param_grid=params, n_jobs=-1, verbose=2)
    grid_search.fit(X=X_train, y=y_train)
    catboost_model=grid_search.best_estimator_

    print(grid_search.best_params_)

    baseline_model = load(baseline_model_path)

    y_pred_baseline = baseline_model.predict(X_test)
    y_pred_catboost = catboost_model.predict(X_test)    # Предсказания с модели CatBoost

    mae_baseline = mean_absolute_error(y_test, y_pred_baseline)
    mae_catboost = mean_absolute_error(y_test, y_pred_catboost)

    print(grid_search.score(X_test,y_test))
    print(f"Модель CatBoost ")
    print("-------------------------------")
    print(f"средння абсолютная ошибка линейной регресси: {mae_baseline:.4f}")
    print(f"средняя абсолютная ошбика cat boost: {mae_catboost:.4f}")

    dump(catboost_model, output_model_joblib_path)