# Подготовка производимая в еде, только без мишуры
# Импортируем все необходимые библиотеки
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import yaml


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', '-id', type=str, default='data/raw/', required=False, help='path to input data directory')
    parser.add_argument('--output_dir', '-od', type=str, default='data/prepared/', required=False, help='path to save prepared data')
    parser.add_argument('--params', '-p', type=str, default='params.yaml', required=False, help='file with dvc stage params')
    parser.add_argument('--user_file', '-uf', type=str, required=True, help='Path to users data file')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    # Считываем данные, выводим размер таблицы
    scores = pd.read_csv(args.user_file)
    with open(args.params, 'r') as f:
        params_all = yaml.safe_load(f)
        params = params_all['data_preparation']

    # Исправим данные в столбце
    real_n_student = scores['classroom'].value_counts()
    for i in range(scores.shape[0]):
        scores.at[i, 'n_student'] = real_n_student[scores.loc[i, 'classroom']]

    scores['f_lunch'] = scores['lunch'].replace(
        ['Qualifies for reduced/free lunch', 'Does not qualify'], [0, 1])
    scores['f_teaching_method'] = scores['teaching_method'].replace(
        ['Standard', 'Experimental'], [0, 1])
    scores['f_school_setting'] = scores['school_setting'].replace(
        ['Rural', 'Urban', 'Suburban'], [0, 1, 2])
    scores['f_school_type'] = scores['school_type'].replace(
        ['Public', 'Non-public'], [0, 1])

    # 3. Средний балл класса за предварительный тест
    scores['f_class_mean'] = scores[['classroom','pretest']].groupby(['classroom']).transform('mean')

    # Экспортируем данные для обучения модели линейной регрессии
    scores = scores.select_dtypes(include=['number'])

    X, y = scores.drop(columns='posttest'), scores['posttest']

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        train_size=params.get('train_test_ratio'),
                                                        random_state=params.get('random_state'))
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                        train_size=params.get('train_val_ratio'),
                                                        random_state=params.get('random_state'))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    X_full_name = output_dir / 'X_full.csv'
    y_full_name = output_dir / 'y_full.csv'
    X_train_name = output_dir / 'X_train.csv'
    y_train_name = output_dir / 'y_train.csv'
    X_test_name = output_dir / 'X_test.csv'
    y_test_name = output_dir / 'y_test.csv'
    X_val_name = output_dir / 'X_val.csv'
    y_val_name = output_dir / 'y_val.csv'
    
    X.to_csv(X_full_name, index=False)
    y.to_csv(y_full_name, index=False)
    X_train.to_csv(X_train_name, index=False)
    y_train.to_csv(y_train_name, index=False)
    X_test.to_csv(X_test_name, index=False)
    y_test.to_csv(y_test_name, index=False)
    X_val.to_csv(X_val_name, index=False)
    y_val.to_csv(y_val_name, index=False)

    # X.to_csv(args.full_features_path, index=False)
    # Y.to_csv(args.full_target_path, index=False)
    # X_train.to_csv(args.train_features_path, index=False)
    # y_train.to_csv(args.train_target_path, index=False)
    # X_test.to_csv(args.test_features_path, index=False)
    # Y_test.to_csv(args.test_target_path, index=False)
