import pandas as pd
import argparse
import warnings
from pathlib import Path
import yaml
import numpy as np
from sklearn.model_selection import train_test_split

warnings.simplefilter(action='ignore', category=FutureWarning)

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
    # Разделение 'name' на 'brand' и 'model'
    combined_df[['brand', 'model']] = combined_df['name'].str.extract(r'(\w+)\s(.+)', expand=True)
    combined_df.drop('name', axis=1, inplace=True)

    # Преобразование 'owner' в числовой вид
    owner_mapping = {'Test Drive Car': 0, 'First Owner': 1, 'Second Owner': 2, 'Third Owner': 3,
                     'Fourth & Above Owner': 4}
    combined_df['owner'] = combined_df['owner'].map(owner_mapping)

    # Разделение данных в столбце 'torque'
    regex = r'(\d+\.?\d*)\s?(?:Nm|nm|kgm)?.*?(\d+\.?\d*)(?:-\d+)?\s?(rpm|RPM)?'
    torque_df = combined_df['torque'].str.extract(regex, expand=True)
    combined_df['Nm'] = pd.to_numeric(torque_df[0], errors='coerce')  # Значение момента, Nm
    combined_df['rpm'] = pd.to_numeric(torque_df[1], errors='coerce')  # Среднее значение rpm
    combined_df.drop('torque', axis=1, inplace=True)

    # Преобразование значений в числовой вид для 'max_power', 'engine', 'mileage'
    for col in ['max_power', 'engine', 'mileage']:
        combined_df[col] = combined_df[col].astype(str).str.extract('(\d+\.?\d*)').astype(float)

    # Добавление нового признака 'price_category'
    # combined_df['price_category'] = np.where(combined_df['selling_price'] > 1_000_000, 'Premium', 'Casual')

    # Определение страны-производителя по бренду
    country_brand_mapping = {
        'India': 'Maruti, Tata, Mahindra, Force, Ambassador, Ashok',
        'Japan': 'Datsun, Honda, Toyota, Nissan, Mitsubishi, Lexus, Isuzu',
        'South Korea': 'Hyundai, Kia, Daewoo',
        'USA': 'Chevrolet, Ford, Jeep',
        'Europe': 'Benz, Audi, BMW, Volkswagen, Opel, Skoda, Renault, Fiat, Volvo',
        'UK': 'Land, MG, Jaguar'
    }

    def get_country_by_brand(brand):
        for country, brands in country_brand_mapping.items():
            if brand in brands.split(', '):
                return country
        return None

    combined_df['country'] = combined_df['brand'].apply(get_country_by_brand)

    # Добавление признака 'years_old' для возраста автомобиля
    current_year = 2025
    combined_df['years_old'] = current_year - combined_df['year']

    # Пробег в зависимости от возраста
    combined_df['km_per_year'] = combined_df['km_driven'] / (combined_df['years_old'] + 0.000001)

    # Пробег в зависимости от количества владельцев
    combined_df['km_per_owner'] = combined_df['km_driven'] / (combined_df['owner'] + 0.000001)

    # Удельный расход топлива
    combined_df['fuel_consumption'] = combined_df['engine'] / (combined_df['mileage'] + 0.000001)

    # Удаление двух автомобилей с наивысшей ценой
    combined_df = combined_df.sort_values(by='selling_price').iloc[:-2]

    # Удаление четырех автомобилей с наивысшим пробегом
    combined_df = combined_df.sort_values(by='km_driven').iloc[:-4]

    print(combined_df.dtypes)
    print("==============================================")
    print(combined_df.isnull().sum())
    return combined_df

def one_hot_encoding(combined_df: pd.DataFrame) -> pd.DataFrame:
    data = {
        'fuel': ['Diesel', 'CNG', 'Petrol', 'LPG'],
        'seller_type': ['Individual', 'Dealer', 'Trustmark Dealer', 'Individual'],
        'transmission': ['Manual', 'Automatic', 'Manual', 'Automatic'],
        # 'price_category': ['Casual', 'Premium'],
        'country': ['India', 'USA', 'Japan', 'South Korea']
    }
    # Применяем One-Hot Encoding
    df_one_hot = pd.get_dummies(combined_df, columns=['fuel', 'seller_type', 'transmission', 'country'])
    df_one_hot = df_one_hot.replace({True: 1, False: 0})
    return df_one_hot


def data_normalization(combined_df: pd.DataFrame) -> pd.DataFrame:
    # km_driven,owner,mileage,engine,max_power,seats,Nm,rpm,years_old,km_per_year,km_per_owner,fuel_consumption
    combined_df['log_km_driven'] = np.log(combined_df['km_driven'] + 10 ** -6)
    combined_df['log_owner'] = np.log(combined_df['owner'] + 10 ** -6)
    combined_df['log_mileage'] = np.log(combined_df['mileage'] + 10 ** -6)
    combined_df['log_engine'] = np.log(combined_df['engine'] + 10 ** -6)
    combined_df['log_max_power'] = np.log(combined_df['max_power'] + 10 ** -6)
    combined_df['log_seats'] = np.log(combined_df['seats'] + 10 ** -6)
    combined_df['log_Nm'] = np.log(combined_df['Nm'] + 10 ** -6)
    combined_df['log_rpm'] = np.log(combined_df['rpm'] + 10 ** -6)
    combined_df['log_years_old'] = np.log(combined_df['years_old'] + 10 ** -6)
    combined_df['log_km_per_year'] = np.log(combined_df['km_per_year'] + 10 ** -6)
    combined_df['log_km_per_owner'] = np.log(combined_df['km_per_owner'] + 10 ** -6)
    combined_df['log_fuel_consumption'] = np.log(combined_df['fuel_consumption'] + 10 ** -6)
    combined_df = combined_df.drop(['km_driven', 'owner', 'mileage', 'engine', 'max_power', 'seats', 'Nm', 'rpm', 'years_old', 'km_per_year', 'km_per_owner', 'fuel_consumption'], axis=1)
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
        combined_df = clean_data(combined_df)
        combined_df = one_hot_encoding(combined_df)
        combined_df = data_normalization(combined_df)
        combined_df['selling_price'] = np.log(combined_df['selling_price'] + 10 ** -6)
        X, y = combined_df.drop(["selling_price", "year", 'brand', 'model', 'log_km_per_year'], axis=1), combined_df['selling_price']

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
        print("Data_preparation.py выполнен успешно!")

    except Exception as e:
        print(f"Ошибка во время обработки данных: {e}")