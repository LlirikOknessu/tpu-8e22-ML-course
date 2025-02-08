import pandas as pd
import argparse
from pathlib import Path
import yaml
import numpy as np
from sklearn.model_selection import train_test_split
import devtools
from scipy.stats.mstats import winsorize


def parser_args_for_sac():
    parser = argparse.ArgumentParser(description='Paths parser')
    parser.add_argument('--input_dir', '-id', type=str, default='data/raw/',
                        required=False, help='path to input data directory')
    parser.add_argument('--output_dir', '-od', type=str, default='data/prepared/',
                        required=False, help='path to save prepared data')
    parser.add_argument('--params', '-p', type=str, default='params.yaml', required=False,
                        help='file with dvc stage params')
    return parser.parse_args()


def to_categorical(df: pd.DataFrame):
    df.experience_level = pd.Categorical(df.experience_level)
    df = df.assign(experience_level=df.experience_level.cat.codes)
    df.employment_type = pd.Categorical(df.employment_type)
    df = df.assign(employment_type=df.employment_type.cat.codes)
    df.employee_residence = pd.Categorical(df.employee_residence)
    df = df.assign(employee_residence=df.employee_residence.cat.codes)
    df.company_size = pd.Categorical(df.company_size)
    df = df.assign(company_size=df.company_size.cat.codes)
    df.company_location = pd.Categorical(df.company_location)
    df = df.assign(company_location=df.company_location.cat.codes)
    df.job_title = pd.Categorical(df.job_title)
    df = df.assign(job_title=df.job_title.cat.codes)
    df.work_year = pd.Categorical(df.work_year)
    df = df.assign(work_year=df.work_year.cat.codes)
    df.remote_ratio = pd.Categorical(df.remote_ratio)
    df = df.assign(remote_ratio=df.remote_ratio.cat.codes)
    return df


def titles_reduction(x) -> str:
    if x.find("Data Science") >= 0 or x.find("Data Scientist") >= 0:
        return 'Data Scientist'
    elif x.find("Analyst") >= 0 or x.find("Analytics") >= 0:
        return 'Data Analyst'
    elif x.find("ML") >= 0 or x.find("Machine Learning") >= 0:
        return 'Machine Learning Engineer'
    elif x.find("Data Engineer") >= 0 or x.find("Data Engineering") >= 0:
        return 'Data Engineer'
    else:
        return 'AI related'


def res(x) -> str:
    if x == "US":
        return "US"
    else:
        return "Other"

def clean_data(data: pd.DataFrame) -> pd.DataFrame:

    #delete NaN useges median fuller
    data['Adult Mortality'] = data['Adult Mortality'].fillna(data['Adult Mortality'].mean())
    data['Life expectancy'] = data['Life expectancy'].fillna(data['Life expectancy'].mean())
    data['Alcohol'] = data['Alcohol'].fillna(data['Alcohol'].mean())
    data['Hepatitis B'] = data['Hepatitis B'].fillna(data['Hepatitis B'].mean())
    data['BMI'] = data['BMI'].fillna(data['BMI'].mean())
    data['Polio'] = data['Polio'].fillna(data['Polio'].mean())
    data['Total expenditure'] = data['Total expenditure'].fillna(data['Total expenditure'].mean())
    data['Diphtheria'] = data['Diphtheria'].fillna(data['Diphtheria'].mean())
    data['GDP'] = data['GDP'].fillna(data['GDP'].mean())
    data['thinness 10-19 years'] = data['thinness 10-19 years'].fillna(
        data['thinness 10-19 years'].mean())
    data['thinness 5-9 years'] = data['thinness 5-9 years'].fillna(data['thinness 5-9 years'].mean())
    data['Population'] = data['Population'].fillna(data['Population'].mean())
    data['Income composition of resources'] = data['Income composition of resources'].fillna(
        data['Income composition of resources'].mean())
    data['Schooling'] = data['Schooling'].fillna(data['Schooling'].mean())

    #drop redundant data
    data.drop(data[(data['Measles'] > 1000)].index, inplace=True)
    data.drop(data[(data['under-five deaths'] > 1000)].index, inplace=True)
    data.drop(data[(data['Population'] < 300000)].index, inplace=True)

    #winsorize data
    data['Adult Mortality'] = winsorize(data['Adult Mortality'], limits=(0, 0.0314))
    data['infant deaths'] = winsorize(data['infant deaths'], limits=(0, 0.109))
    data['Alcohol'] = winsorize(data['Alcohol'], limits=(0, 0.1))
    data['percentage expenditure'] = winsorize(data['percentage expenditure'], limits=(0, 0.1333))
    data['Hepatitis B'] = winsorize(data['Hepatitis B'], limits=(0.10725, 0))
    data['Measles'] = winsorize(data['Measles'], limits=(0, 0.1493))
    data['BMI'] = winsorize(data['BMI'], limits=(0, 0))
    data['under-five deaths'] = winsorize(data['under-five deaths'], limits=(0, 0.1232))
    data['Polio'] = winsorize(data['Polio'], limits=(0.0936, 0))
    data['Total expenditure'] = winsorize(data['Total expenditure'], limits=(0, 0.0214))
    data['Diphtheria'] = winsorize(data['Diphtheria'], limits=(0.0942, 0))
    data['HIV/AIDS'] = winsorize(data['HIV/AIDS'], limits=(0, 0.1972))
    data['GDP'] = winsorize(data['GDP'], limits=(0, 0.1131))
    data['Population'] = winsorize(data['Population'], limits=(0, 0.061))
    data['thinness 10-19 years'] = winsorize(data['thinness 10-19 years'], limits=(0, 0.0231))
    data['thinness 5-9 years'] = winsorize(data['thinness 5-9 years'], limits=(0, 0.0285))
    data['Income composition of resources'] = winsorize(data['Income composition of resources'], limits=(0.0456, 0))
    data['Schooling'] = winsorize(data['Schooling'], limits=(0.0326, 0.0024))
    data['Life expectancy'] = winsorize(data['Life expectancy'], limits=(0.0202, 0))

    #drop columns
    data = data.drop(columns=['Country', 'Status', 'infant deaths'], axis=1)

    #concat, drop and rename columns
    data['thinness 5-9 years'] = data['thinness 5-9 years'] + data['thinness 10-19 years']
    data = data.drop('thinness 10-19 years', axis=1)
    data.rename(columns={'thinness 5-9 years': 'thinness 5-19 years'}, inplace=True)
    name = ['Year', 'Adult Mortality', 'Alcohol', 'percentage expenditure', 'Hepatitis B', 'Measles',
            'BMI', 'under-five deaths', 'Polio', 'Total expenditure', 'Diphtheria', 'HIV/AIDS',
            'GDP', 'Population', 'thinness 5-19 years', 'Income composition of resources', 'Schooling']

    #log_data
    for i, c in enumerate(name):
        data[c] = np.log(data[c] + 10 ** -6)

    return data


if __name__ == '__main__':
    args = parser_args_for_sac()
    with open(args.params, 'r') as f:
        params_all = yaml.safe_load(f)
    params = params_all['data_preparation']

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(exist_ok=True, parents=True)

    for data_file in input_dir.glob('*.csv'):
        full_data = pd.read_csv(data_file)
        cleaned_data = clean_data(data=full_data)
        X, y = cleaned_data.drop("Life expectancy", axis=1), cleaned_data['Life expectancy']
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            train_size=params.get('train_test_ratio'),
                                                            random_state=params.get('random_state'))
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                          train_size=params.get('train_val_raitio'),
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
