# Импортируем нужные библиотеки
import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

def data_split(x, y):
    return train_test_split(x, y, test_size=0.2, train_size=0.8, random_state=2)

def test_split(x, y):
    return train_test_split(x, y, test_size=0.4, train_size=0.6, random_state=2)

if __name__ == '__main__':
    # Загружеаем данные для обучения
    df = pd.read_csv('data/prepared/prepared_data.csv')
    X, Y = df.drop(columns='posttest'), df['posttest']

    X_train, X_test, Y_train, Y_test = data_split(X, Y)
    X_val, X_test, Y_val, Y_test = test_split(X_test, Y_test)

    # Обучаем модель
    model = LinearRegression()
    model.fit(X_train, Y_train)

    # Заполняем датафрейм
    df = X_test
    df['posttest_pred'] = model.predict(X_test)
    df['posttest_real'] = Y_test
    df['error'] = df['posttest_real'] - df['posttest_pred']
    
    # Считаем среднеквадратичное отклонение
    mse = np.mean(df['error'] ** 2)
    print('MSE (hypotetical dataset): ', mse)

    # Считаем метрику R2
    r2 = r2_score(df.posttest_real, df.posttest_pred)
    print('R2: (hypotetical dataset): ', r2)