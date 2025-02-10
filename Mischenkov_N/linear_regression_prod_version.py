import pandas as pd
import argparse
from pathlib import Path
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from joblib import dump

LINEAR_MODELS_MAPPER = {'Ridge': Ridge,
                        'LinearRegression': LinearRegression}


def parser_args_for_sac():
    parser = argparse.ArgumentParser(description='Paths parser')
    parser.add_argument('--input_dir', '-id', type=str, default='data/prepared/',
                        required=False, help='path to input data directory')
    parser.add_argument('--output_dir', '-od', type=str, default='data/models/',
                        required=False, help='path to save prepared data')
    parser.add_argument('--model_name', '-mn', type=str, default='LR', required=False,
                        help='name for the model')
    return parser.parse_args()


if __name__ == '__main__':
    args = parser_args_for_sac()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(exist_ok=True, parents=True)

    output_model_path = output_dir / (args.model_name + '_prod.csv')
    output_model_joblib_path = output_dir / (args.model_name + '_prod.joblib')

    X_full_path = input_dir / 'X_full.csv'
    y_full_path = input_dir / 'y_full.csv'
    X_full = pd.read_csv(X_full_path)
    y_full = pd.read_csv(y_full_path)

    print(f"Подготовка модели {args.model_name} для финального обучения...")# Параметры
    if args.model_name == "Ridge":
        reg = LINEAR_MODELS_MAPPER.get(args.model_name)(alpha=1.0)  # Используем параметр alpha
    else:  # Для LinearRegression параметров не передаем
        reg = LINEAR_MODELS_MAPPER.get(args.model_name)()

    reg.fit(X_full, y_full)
    # print("Обучение завершено на полном наборе данных.")

    # Оценка на тех же данных (поскольку это финальная версия)
    predicted_values = np.squeeze(reg.predict(X_full))
    mae = mean_absolute_error(y_full, predicted_values)
    r2_score = reg.score(X_full, y_full)

    print(f"\nProduction модель метрики (обучение на всех данных):")
    print(f"MAE: {mae:.4f}")
    print(f"R² Score: {r2_score:.4f}")

    intercept = reg.intercept_
    coefficients = reg.coef_
    intercept = pd.Series(intercept, name='intercept')
    coefficients = pd.Series(coefficients[0], name='coefficients')  # Преобразуем в Series

    out_model = pd.DataFrame([coefficients, intercept])
    out_model.to_csv(output_model_path, index=False)
    print(f"Коэффициенты модели сохранены в {output_model_path}")

    dump(reg, output_model_joblib_path)
    print(f"Production модель сохранена в {output_model_joblib_path}")
