import pandas as pd
import argparse
from pathlib import Path
import yaml
import numpy as np
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from joblib import dump
import matplotlib.pyplot as plt

TREES_MODELS_MAPPER = {
    'DecisionTree': tree.DecisionTreeRegressor,
    'RandomForest': RandomForestRegressor,
    'ExtraTree': ExtraTreesRegressor
}

TREES_MODELS_BEST_PARAMETERS = {
    'DecisionTree': {'max_depth': 7, 'min_samples_leaf': 4, 'min_samples_split': 3, 'splitter': 'best'},
    'RandomForest': {'max_depth': 6, 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 15},
    'ExtraTree': {'max_depth': 7, 'min_samples_leaf': 2, 'min_samples_split': 10, 'n_estimators': 15}
}


def parser_args_for_sac():
    parser = argparse.ArgumentParser(description='Paths parser')
    parser.add_argument('--input_dir', '-id', type=str, default='data/prepared/',
                        required=False, help='path to input data directory')
    parser.add_argument('--output_dir', '-od', type=str, default='data/models/',
                        required=False, help='path to save prepared data')
    parser.add_argument('--model_name', '-mn', type=str, default='DecisionTree',
                        required=False, help='model name (DecisionTree, RandomForest, ExtraTree)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parser_args_for_sac()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(exist_ok=True, parents=True)
    output_model_path = output_dir / (args.model_name + '_prod.jpg')
    output_model_joblib_path = output_dir / (args.model_name + '_prod.joblib')

    X_train_name = input_dir / 'X_train.csv'
    y_train_name = input_dir / 'y_train.csv'
    X_test_name = input_dir / 'X_test.csv'
    y_test_name = input_dir / 'y_test.csv'

    X_train = pd.read_csv(X_train_name)
    y_train = pd.read_csv(y_train_name)
    X_test = pd.read_csv(X_test_name)
    y_test = pd.read_csv(y_test_name)

    X_full = pd.concat([X_train, X_test], axis=0)
    y_full = pd.concat([y_train, y_test], axis=0)

    best_params = TREES_MODELS_BEST_PARAMETERS.get(args.model_name)

    reg = TREES_MODELS_MAPPER.get(args.model_name)(**best_params)

    if isinstance(reg, (RandomForestRegressor, ExtraTreesRegressor)):
        y_train = np.ravel(y_train.values)

    reg = reg.fit(X_train, y_train)

    dump(reg, output_model_joblib_path)
    print(f"Production модель '{args.model_name}' сохранена в {output_model_joblib_path}")

    # строим графическое представление
    if isinstance(reg, tree.DecisionTreeRegressor):
        fig = plt.figure(figsize=(60, 25))
        tree.plot_tree(
            reg,
            feature_names=X_train.columns,
            filled=True
        )
        fig.savefig(output_model_path)
        print(f"Графическое представление дерева решений сохранено в {output_model_path}")

    y_pred_full = reg.predict(X_full)

    mae_full = mean_absolute_error(y_full, y_pred_full)
    r2_full = r2_score(y_full, y_pred_full)

    print(f"MAE:      {mae_full:.4f}")
    print(f"R² Score: {r2_full:.4f}")

