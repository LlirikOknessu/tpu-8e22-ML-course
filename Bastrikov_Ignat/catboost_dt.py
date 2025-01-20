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
import matplotlib.pyplot as plt
import seaborn as sns

def parser_args_for_sac():
    parser = argparse.ArgumentParser(description='Paths parser')
    parser.add_argument('--input_dir', '-id', type=str, default='data/prepared/',
                        required=False, help='path to input data directory')
    parser.add_argument('--output_dir', '-od', type=str, default='data/models/',
                        required=False, help='path to save prepared data')
    parser.add_argument('--baseline_model', '-bm', type=str, default='data/models/LinearRegression_prod.joblib',
                        required=False, help='path to linear regression prod version')
    parser.add_argument('--model_name', '-mn', type=str, default='LR', required=False,
                        help='name of the model being trained')
    parser.add_argument('--params', '-p', type=str, default='params.yaml', required=False,
                        help='file with dvc stage params')
    return parser.parse_args()

def plot_feature_importance(model, feature_names, output_path):

    importance = model.get_feature_importance(prettified=True)
    importance_df = pd.DataFrame(importance)
    importance_df.set_index('Feature Id', inplace=True)

    importance_df = importance_df.sort_values(by='Importances', ascending=False)

    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importances', y=importance_df.index, data=importance_df, palette='viridis')
    plt.title('Feature Importance')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()

    output_path = Path(output_dir) / "feature_importance.png"

    plt.savefig(output_path)
    plt.close()
    print(f"График важности признаков сохранен в {output_path}")

def calculate_mean_deviation(mae_res, outliers, genres, types):
    mean_deviation_results = []
    mean_deviation_results.append({'category': '=== mae_res ===', 'mean_deviation': None, 'count': None})

    # для mae_res
    for genre in genres:
        genre_mask = mae_res[genre] == 1
        if genre_mask.sum() > 0:
            mean_dev = mae_res.loc[genre_mask, 'difference'].mean()
            mean_deviation_results.append({'category': genre, 'mean_deviation': mean_dev, 'count': genre_mask.sum()})

    for anime_type in types:
        type_mask = mae_res[anime_type] == 1
        if type_mask.sum() > 0:
            mean_dev = mae_res.loc[type_mask, 'difference'].mean()
            mean_deviation_results.append({'category': anime_type, 'mean_deviation': mean_dev, 'count': type_mask.sum()})

    mean_deviation_results.append({'category': '=== OUTLIERS ===', 'mean_deviation': None, 'count': None})

    # для outliers
    for genre in genres:
        genre_mask = outliers[genre] == 1
        if genre_mask.sum() > 0:
            mean_dev = outliers.loc[genre_mask, 'difference'].mean()
            mean_deviation_results.append({'category': genre, 'mean_deviation': mean_dev, 'count': genre_mask.sum()})

    for anime_type in types:
        type_mask = outliers[anime_type] == 1
        if type_mask.sum() > 0:
            mean_dev = outliers.loc[type_mask, 'difference'].mean()  # Среднее отклонение
            mean_deviation_results.append({'category': anime_type, 'mean_deviation': mean_dev, 'count': type_mask.sum()})

    # Создаём DataFrame из результатов
    mean_deviation_results_df = pd.DataFrame(mean_deviation_results)
    mean_deviation_results_df.to_csv(".\data\prepared\mean_deviation.csv", index=False)
    print(f"\nСреднее отклонение по жанрам и типам сохранено")

    # export
    # outliers.to_csv(".\data\prepared\outliers.csv", index=False)
    #print("Outliers для каждого столбика сохранен в data\prepared")
    #mae_res.to_csv(".\data\prepared\mae_res.csv", index=False)
    #print("Mae для каждого столбика сохранен в data\prepared")


if __name__ == '__main__':
    args = parser_args_for_sac()

    with open(args.params, 'r') as f:
        params_all = yaml.safe_load(f)
    params = params_all['catboost']['CatBoost']

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

    catboost_model = CatBoostRegressor(random_state=42, verbose=False)

    grid_search = GridSearchCV(estimator=catboost_model,
                               param_grid=params,
                               scoring='neg_mean_absolute_error',
                               cv=5,
                               n_jobs=-1,
                               verbose=1)

    grid_search.fit(X=X_train, y=y_train)

    baseline_model = load(baseline_model_path)

    y_pred_baseline = baseline_model.predict(X_test)
    y_pred_catboost = grid_search.predict(X_test)    # Предсказания с модели CatBoost

    mae_baseline = mean_absolute_error(y_test, y_pred_baseline)
    mae_catboost = mean_absolute_error(y_test, y_pred_catboost)

    mae_res = pd.DataFrame({
        'true_values': y_test,
        'predicted_values': y_pred_catboost,
        'difference': abs(y_test - y_pred_catboost)
    })
    outliers = mae_res[mae_res["difference"] > 1]


    print(f"Модель CatBoost ({args.model_name})")
    print("-------------------------------")
    print(f"Score: {grid_search.best_score_:.4f}")
    print(f"Лучшие параметры: {grid_search.best_params_}")
    print(f"Baseline MAE: {mae_baseline:.4f}")
    print(f"Model MAE: {mae_catboost:.4f}")
    print('outliers count:', outliers.shape[0])
    print('mae_res count:', mae_res.shape[0])
    print('% of outliers', (outliers.shape[0] / mae_res.shape[0]) * 100)

    print('==============================test==============================')
    # Create extra files
    genres = ['Action/Adventure', 'Fantasy/Supernatural', 'Comedy', 'Drama/Romance', 'Science Fiction', 'Psychological/Thriller', 'Other/Uncategorized']
    types = ['type_Movie', 'type_Music', 'type_ONA', 'type_OVA', 'type_Special', 'type_TV']

    existing_columns = [col for col in genres if col in X_test.columns]
    existing_types = [col for col in types if col in X_test.columns]

    outliers.to_csv(".\data\prepared\outliers.csv", index=False)
    mae_res.to_csv(".\data\prepared\mae_res.csv", index=False)
    print("Mae_res в общем виде сохранен в data\prepared")
    print("Outliers в общем виде в data\prepared")

    mae_res_genres_and_types = pd.concat([mae_res, X_test[genres], X_test[types]], axis=1)
    outliers_genres_and_types = pd.concat([outliers, X_test[genres], X_test[types]], axis=1)
    outliers_genres_and_types.dropna(inplace=True)
    mae_res_genres_and_types.dropna(inplace=True)
    calculate_mean_deviation(mae_res_genres_and_types, outliers_genres_and_types, genres, types)
    # ========================== feature importance =======================================
    feature_names = X_train.columns.tolist()
    plot_feature_importance(grid_search.best_estimator_, feature_names, output_dir)
    # =====================================================================================
    # export
    outliers_genres_and_types.to_csv(".\data\prepared\outliers_genres_and_types.csv", index=False)
    mae_res_genres_and_types.to_csv(".\data\prepared\mae_res_genres_and_types.csv", index=False)

    print("Outliers для genres_and_types сохранен в data\prepared")
    print("Mae для genres_and_types сохранен в data\prepared")



    dump(grid_search.best_estimator_, output_model_joblib_path)