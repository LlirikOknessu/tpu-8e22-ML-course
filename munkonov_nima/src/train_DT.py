import pandas as pd
from pathlib import Path
import yaml
import numpy as np
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error
# from joblib import dump, load
import random
from sklearn.model_selection import GridSearchCV
import pickle

TREES_MODELS_MAPPER = {'DecisionTree': tree.DecisionTreeRegressor,
                       'RandomForest': RandomForestRegressor,
                       'ExtraTree': ExtraTreesRegressor}


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_features_path', '-fp', type=Path, required=True)
    parser.add_argument('--train_target_path', '-tp', type=Path, required=True)
    parser.add_argument('--output_model_path', '-omp', type=Path, required=True)
    parser.add_argument('--params', '-p', type=Path, required=True)

    # parser.add_argument('--input_dir', '-id', type=str, default='data/prepared/',
    #                     required=False, help='path to input data directory')
    # parser.add_argument('--output_dir', '-od', type=str, default='data/models/',
    #                     required=False, help='path to save prepared data')
    # parser.add_argument('--model_name', '-mn', type=str, default='LR', required=False,
    #                     help='file with dvc stage params')
    # parser.add_argument('--params', '-p', type=str, default='params.yaml', required=False,
    #                     help='file with dvc stage params')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    with open(args.params, 'r') as f:
        params_all = yaml.safe_load(f)
    params = params_all['decision_tree']

    X_train = pd.read_csv(args.train_features_path)
    y_train = pd.read_csv(args.train_target_path)

    # random.seed(42)
    # decision_tree_model = TREES_MODELS_MAPPER.get(args.model_name)()
    decision_tree_model = tree.DecisionTreeRegressor()
    decision_tree_regressor = GridSearchCV(decision_tree_model, params['DecisionTree'])

    if (isinstance(decision_tree_model, RandomForestRegressor) 
        or isinstance(decision_tree_model, ExtraTreesRegressor)):
        y_train = np.ravel(y_train.values)

    decision_tree_regressor = decision_tree_regressor.fit(X_train, y_train)

    print(decision_tree_regressor.best_params_)

    pickle.dump(decision_tree_regressor, open(args.output_model_path, 'wb'))