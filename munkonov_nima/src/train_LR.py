# Импортируем нужные библиотеки
import pandas as pd
from sklearn.linear_model import LinearRegression
from pathlib import Path
import pickle


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_features_path', '-fp', type=Path, required=True)
    parser.add_argument('--train_target_path', '-tp', type=Path, required=True)
    parser.add_argument('--output_model_path', '-omp', type=Path, required=True)
    return parser.parse_args()


if __name__ == '__main__':
    # Интерпретируем аргументы
    args = parse_args()

    # Загружеаем данные для обучения
    X_train = pd.read_csv(args.train_features_path)
    Y_train = pd.read_csv(args.train_target_path)

    # Обучаем модель
    model = LinearRegression()
    model.fit(X_train, Y_train)

    # Сохраняем модель
    pickle.dump(model, open(args.output_model_path, 'wb'))
