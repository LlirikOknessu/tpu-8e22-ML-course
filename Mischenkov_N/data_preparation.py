import pandas as pd
import argparse
from pathlib import Path
import yaml
import numpy as np
from sklearn.model_selection import train_test_split

def parser_args_for_sac():
    parser = argparse.ArgumentParser(description='Paths parser')
    parser.add_argument('--input_dir', '-id', type=str, default='data/raw/', required=False, help='path to input data directory')
    parser.add_argument('--output_dir', '-od', type=str, default='data/prepared/', required=False, help='path to save prepared data')
    parser.add_argument('--params', '-p', type=str, default='params.yaml', required=False, help='file with dvc stage params')
    return parser.parse_args()


def combine_dataframes(df1: pd.DataFrame, df2: pd.DataFrame, df3: pd.DataFrame) -> pd.DataFrame:
    duplicates = pd.merge(df2[['name']], df3[['name']], on='name')

    # Фильтруем оригинальные данные дубликатов из первого датафрейма
    duplicates_from_df2 = df2[df2['name'].isin(duplicates['name'])]

    # Объединяем данные
    combined_df = pd.concat([duplicates_from_df2, df3], axis=0)

    # Список колонок для заполнения пропусков
    cols_to_fill = ['mileage', 'engine', 'max_power', 'torque', 'seats']

    # Заполнение пропусков внутри групп
    for col in cols_to_fill:
        combined_df[col] = combined_df.groupby('name')[col].transform(
            lambda group: group.bfill().ffill()
        )

    # Удаление оставшихся пропусков
    combined_df = combined_df.dropna()
    return combined_df


def clean_data(combined_df: pd.DataFrame) -> pd.DataFrame:
    return combined_df


if __name__ == '__main__':
    args = parser_args_for_sac()
    try:
        with open(args.params, 'r') as f:
            params_all = yaml.safe_load(f)
            params = params_all['data_preparation']

        input_dir = Path(args.input_dir)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        try:
            df1 = pd.read_csv(input_dir / 'car_data.csv')
            print(f"Прочитан файл 1, строк: {len(df1)}")
            df2 = pd.read_csv(input_dir / 'CAR_DETAILS_FROM_CAR_DEKHO.csv')
            print(f"Прочитан файл 2, строк: {len(df2)}")
            df3 = pd.read_csv(input_dir / 'Car_details_v3.csv')
            print(f"Прочитан файл 3, строк: {len(df3)}")
        except:
            print('Ошибка при чтении файлов')

        combined_df = combine_dataframes(df1, df2, df3)
        X, y = combined_df.drop("selling_price", axis=1), combined_df['selling_price']
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            train_size=params.get('train_test_ratio'),
                                                            random_state=params.get('random_state'))
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                            train_size=params.get('train_val_ratio'),
                                                            random_state=params.get('random_state'))

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

    except Exception as e:
        print(f"Ошибка во время обработки данных: {e}")