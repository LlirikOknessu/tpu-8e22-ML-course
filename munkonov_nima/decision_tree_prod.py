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
    output_model_path = output_dir / (args.model_name + '_prod.pdf')
    output_model_joblib_path = output_dir / (args.model_name + '_prod.joblib')

    X_full_name = input_dir / 'X_full.csv'
    y_full_name = input_dir / 'y_full.csv'

    X_full = pd.read_csv(X_full_name)
    y_full = pd.read_csv(y_full_name)

    best_params = pd.read_csv(output_dir / f"{args.model_name}.csv").squeeze()

    reg = TREES_MODELS_MAPPER.get(args.model_name)(**best_params)

    if isinstance(reg, (RandomForestRegressor, ExtraTreesRegressor)):
        y_full = np.ravel(y_full.values)

    reg = reg.fit(X_full, y_full)

    dump(reg, output_model_joblib_path)
    print(f"Production модель '{args.model_name}' сохранена в {output_model_joblib_path}")

    # строим графическое представление
    if isinstance(reg, tree.DecisionTreeRegressor):
        fig = plt.figure(figsize=(60, 25))
        tree.plot_tree(
            reg,
            feature_names=X_full.columns,
            filled=True
        )
        fig.savefig(output_model_path)
        print(f"Графическое представление дерева решений сохранено в {output_model_path}")

    y_pred_full = reg.predict(X_full)

