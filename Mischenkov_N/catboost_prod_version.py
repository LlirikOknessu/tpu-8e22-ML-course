import pandas as pd
import argparse
from pathlib import Path
import yaml
from joblib import dump
from catboost_train import CatBoostRegressor
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
    params = params_all['catboost']['catboost_prod']

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(exist_ok=True, parents=True)
    output_model_joblib_path = output_dir / f"{args.model_name}_prod.joblib"
    output_metrics_path = output_dir / f"{args.model_name}_prod_metrics.json"

    X_full_path = input_dir / 'X_full.csv'
    y_full_path = input_dir / 'y_full.csv'
    X_full = pd.read_csv(X_full_path).squeeze()
    y_full = pd.read_csv(y_full_path).squeeze()

    random.seed(42)

    catboost_model = CatBoostRegressor(
        iterations=params.get('iterations', 100),
        depth=params.get('depth', 6),
        learning_rate=params.get('learning_rate', 0.1),
        l2_leaf_reg=params.get('l2_leaf_reg', 3),
        loss_function=params.get('loss_function', 'RMSE'),
        verbose=params.get('verbose', False),
        random_state=42
    )

    catboost_model.fit(X_full, y_full)
    dump(catboost_model, output_model_joblib_path)

    y_pred_full = catboost_model.predict(X_full)
    mae_full = mean_absolute_error(y_full, y_pred_full)

    metrics = {
        "model_name": args.model_name,
        "iterations": params.get('iterations', 150),
        "depth": params.get('depth', 6),
        "learning_rate": params.get('learning_rate', 0.1),
        "l2_leaf_reg": params.get('l2_leaf_reg', 3),
        "loss_function": params.get('loss_function', 'RMSE'),
        "verbose": params.get('verbose', False),
        "mae_full": mae_full
    }

    mae_full = mean_absolute_error(y_full, y_pred_full)
    mse_full = mean_squared_error(y_full, y_pred_full)
    r2_full = r2_score(y_full, y_pred_full)


    print(f"\nProduction модель метрики (обучение на всех данных):")
    print(f"MAE: {mae_full:.4f}")
    print(f"MSE: {mse_full:.4f}")
    print(f"R² Score: {r2_full:.4f}")

    model_params_metrics = pd.DataFrame({
        'iterations': [catboost_model.get_params().get("iterations")],
        'depth': [catboost_model.get_params().get("depth")],
        'learning_rate': [catboost_model.get_params().get("learning_rate")],
        'l2_leaf_reg': [catboost_model.get_params().get("l2_leaf_reg")],
        'loss_function': [catboost_model.get_params().get("loss_function")],
        'mae': [mae_full],
        'mse': [mse_full],
        'r2_score': [r2_full]
    })

    output_model_path = output_dir / (args.model_name + '_prod.csv')

    # Сохранение в CSV файл
    model_params_metrics.to_csv(output_model_path, index=False)

    print(f"\nПараметры и метрики модели сохранены в {output_model_path}")
    print(f"Production модель сохранена в {output_model_joblib_path}")