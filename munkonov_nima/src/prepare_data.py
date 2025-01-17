# Подготовка производимая в еде, только без мишуры
# Импортируем все необходимые библиотеки
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', '-ip', type=Path, required=True)
    parser.add_argument('--output_path', '-op', type=Path, required=True)
    parser.add_argument('--train_features_path', '-trfp', type=Path, required=True)
    parser.add_argument('--train_target_path', '-trtp', type=Path, required=True)
    parser.add_argument('--test_features_path', '-tefp', type=Path, required=True)
    parser.add_argument('--test_target_path', '-tetp', type=Path, required=True)
    parser.add_argument('--full_features_path', '-ffp', type=Path, required=True)
    parser.add_argument('--full_target_path', '-ftp', type=Path, required=True)
    parser.add_argument('--seed', '-s', type=int, default=42, required=False)
    parser.add_argument('--test_size', '-ts', type=float, default=0.2, required=False)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    # Считываем данные, выводим размер таблицы
    scores = pd.read_csv(args.input_path)

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
    scores.to_csv(args.output_path, index=False)

    X, Y = scores.drop(columns='posttest'), scores['posttest']

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=args.test_size, random_state=args.seed)

    X.to_csv(args.full_features_path, index=False)
    Y.to_csv(args.full_target_path, index=False)
    X_train.to_csv(args.train_features_path, index=False)
    Y_train.to_csv(args.train_target_path, index=False)
    X_test.to_csv(args.test_features_path, index=False)
    Y_test.to_csv(args.test_target_path, index=False)
