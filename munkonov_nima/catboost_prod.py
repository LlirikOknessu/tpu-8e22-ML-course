import pandas as pd
import argparse
from pathlib import Path
from joblib import dump
from catboost import CatBoostRegressor


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

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    best_params = pd.read_csv(output_dir / f"{args.model_name}.csv").squeeze()

    output_dir.mkdir(exist_ok=True, parents=True)
    output_model_joblib_path = output_dir / f"{args.model_name}_prod.joblib"
    output_metrics_path = output_dir / f"{args.model_name}_prod_metrics.json"

    X_full_name = input_dir / 'X_full.csv'
    y_full_name = input_dir / 'y_full.csv'

    X_full = pd.read_csv(X_full_name)
    y_full = pd.read_csv(y_full_name)

    # X_full = pd.concat([X_train, X_test], axis=0).reset_index(drop=True)
    # y_full = pd.concat([y_train, y_test], axis=0).reset_index(drop=True)

    catboost_model = CatBoostRegressor(
        iterations=best_params['iterations'],
        depth=best_params['depth'],
        learning_rate=best_params['learning_rate'],
        l2_leaf_reg=best_params['l2_leaf_reg'],
        loss_function=best_params['loss_function'],
        verbose=bool(best_params['verbose']),
        random_state=42
    )

    catboost_model.fit(X_full, y_full)
    dump(catboost_model, output_model_joblib_path)

    output_model_path = output_dir / (args.model_name + '_prod.csv')

    feature_importance = catboost_model.get_feature_importance()
    feature_names = X_full.columns
    # Display feature importance
    for name, importance in zip(feature_names, feature_importance):
        print(f"Признак: {name}, важность: {importance:.2f}")

    print(f"Production модель сохранена в {output_model_joblib_path}")