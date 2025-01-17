import pandas as pd
from pathlib import Path
import yaml
import numpy as np
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from joblib import load
import pickle
import json
import matplotlib.pyplot as plt


TREES_MODELS_MAPPER = {'DecisionTree': tree.DecisionTreeRegressor,
                       'RandomForest': RandomForestRegressor,
                       'ExtraTree': ExtraTreesRegressor}


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--test_target_path', '-tp', type=Path, required=True)
    parser.add_argument('--test_features_path', '-fp', type=Path, required=True)
    parser.add_argument('--model_path', '-mp', type=Path, required=True)
    parser.add_argument('--output_path', '-op', type=Path, required=True)

    # parser.add_argument('--input_dir', '-id', type=str, default='data/prepared/',
    #                     required=False, help='path to input data directory')
    # parser.add_argument('--input_model', '-im', type=str, default='data/models/',
    #                     required=False, help='path to save prepared data')
    # parser.add_argument('--baseline_model', '-bm', type=str, default='data/models/LinearRegression_prod.joblib',
    #                     required=False, help='path to linear regression prod version')
    # parser.add_argument('--model_name', '-mn', type=str, default='LR', required=False,
    #                     help='file with dvc stage params')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # input_dir = Path(args.input_dir)
    # input_model = Path(args.input_model)
    # baseline_model_path = Path(args.baseline_model)

    # X_val_name = input_dir / 'X_val.csv'
    # y_val_name = input_dir / 'y_val.csv'

    # X_val = pd.read_csv(X_val_name)
    # y_val = pd.read_csv(y_val_name)
    model: tree.DecisionTreeRegressor = pickle.load(open(args.model_path, 'rb'))
    print('model type: ', type(model))
    print('model path: ', args.model_path)
    X_test = pd.read_csv(args.test_features_path)
    Y_test = pd.read_csv(args.test_target_path)

    # reg = load(input_model)

    Y_pred = model.predict(X_test)
    result = {
        'r2': r2_score(Y_test, Y_pred),
        'mse': mean_squared_error(Y_test, Y_pred),
        'mae': mean_absolute_error(Y_test, Y_pred)
    }

    # if isinstance(model, tree.DecisionTreeRegressor):
    fig = plt.figure(figsize=(60,25))
    tree.plot_tree(model,
                   feature_names=X_test.columns,
                   class_names=Y_test.columns,
                   filled=True)
    fig.savefig('decision_tree.jpg')

    with open(args.output_path, 'w') as file:
        json.dump(result, file)

    # y_pred_baseline = np.squeeze(baseline_model.predict(X_test))

    # print(reg.score(X_val, y_val))
    # print("Baseline MAE: ", mean_absolute_error(y_val, y_pred_baseline))
    # print("Model MAE: ", mean_absolute_error(y_val, predicted_values))