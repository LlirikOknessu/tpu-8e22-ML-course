# Импортируем нужные библиотеки
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from pathlib import Path
import pickle


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data_path', '-idp', type=Path, required=True)
    parser.add_argument('--train_features_path', '-trfp', type=Path, required=True)
    parser.add_argument('--test_features_path', '-tefp', type=Path, required=True)
    parser.add_argument('--train_target_path', '-trtp', type=Path, required=True)
    parser.add_argument('--test_target_path', '-tetp', type=Path, required=True)
    parser.add_argument('--output_model_path', '-omp', type=Path, required=True)
    parser.add_argument('--seed', '-s', type=int, default=42, required=False)
    parser.add_argument('--test_size', '-ts', type=float, default=0.2, required=False)
    return parser.parse_args()


if __name__ == '__main__':
    # Интерпретируем аргументы
    args = parse_args()

    # Загружеаем данные для обучения
    df = pd.read_csv(args.input_data_path)
    X, Y = df.drop(columns='posttest'), df['posttest']

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=args.test_size, random_state=args.seed)

    X_train.to_csv(args.train_features_path, index=False)
    Y_train.to_csv(args.train_target_path, index=False)
    X_test.to_csv(args.test_features_path, index=False)
    Y_test.to_csv(args.test_target_path, index=False)

    # Обучаем модель
    model = LinearRegression()
    model.fit(X_train, Y_train)

    # Сохраняем модель
    pickle.dump(model, open(args.output_model_path, 'wb'))
