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
    parser.add_argument('--users_file', '-uf', type=str, required=True, help='Path to users data file')
    return parser.parse_args()


def split_and_explode_genres(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['genre'] = df['genre'].str.split(',')
    df = df.explode('genre').reset_index(drop=True)
    return df


def calculate_genre_stats(df: pd.DataFrame) -> pd.DataFrame:
    df_genres = df.copy()
    df_genres['genre'] = df_genres['genre'].str.split(',')  # Разделяем жанры
    df_genres = df_genres.explode('genre').reset_index(drop=True)  # Разделяем строки для каждого жанра

    # genre_rating = df_genres.groupby('genre')['rating'].mean()
    genre_members = df_genres.groupby('genre')['members'].mean()

    # def avg_genre_rating(genres):
    #     genre_ratings = genre_rating[genres]  # Получаем средний рейтинг жанров
    #     return genre_ratings.mean()

    def avg_members_genre(genres):
        genre_members_ = genre_members[genres]  # Получаем среднее количество участников по жанрам
        return genre_members_.mean()

    # df['avg_rating_genre'] = df['genre'].apply(
    #     lambda genres: avg_genre_rating(genres.split(',')) if isinstance(genres, str) else None
    # )
    df['avg_members_genre'] = df['genre'].apply(
        lambda genres: avg_members_genre(genres.split(',')) if isinstance(genres, str) else None
    )
    return df


def calculate_user_activity(df_users: pd.DataFrame) -> pd.Series:
    df_users.loc[df_users['rating'] == -1, 'rating'] = np.nan
    df_users_grouped = df_users.groupby('anime_id').agg(
        rated_count=('rating', 'count'),
        total_viewers=('rating', 'size')
    )
    df_users_grouped['user_activity'] = (df_users_grouped['rated_count'] / df_users_grouped['total_viewers']) * 100
    return df_users_grouped['user_activity']


def calculate_anime_type_features(df: pd.DataFrame) -> pd.DataFrame:
    df_grouped = df.groupby('type').agg({
        'members': 'mean',
        'rating': 'mean',
        'anime_id': 'count'
    }).reset_index()
    df['anime_count_by_type'] = df['type'].map(df_grouped.set_index('type')['anime_id'])
    # df['avg_rating_by_type'] = df['type'].map(df_grouped.set_index('type')['rating'])
    df['avg_members_by_type'] = df['type'].map(df_grouped.set_index('type')['members'])
    return df


def map_genres_to_categories(genres, genre_categories):
    categories = {category: 0 for category in genre_categories}
    for category, genre_list in genre_categories.items():
        if any(genre in genres for genre in genre_list):
            categories[category] = 1
    return categories


def add_genre_categories(df):
    genre_categories = {
        'Action/Adventure': ['Action', 'Adventure', 'Martial Arts', 'Samurai', 'Military'],
        'Fantasy/Supernatural': ['Fantasy', 'Magic', 'Supernatural', 'Demons', 'Vampire'],
        'Comedy': ['Comedy', 'Parody', 'Slice of Life'],
        'Drama/Romance': ['Drama', 'Romance', 'Josei', 'Shoujo', 'Shounen Ai'],
        'Science Fiction': ['Sci-Fi', 'Space', 'Mecha', 'Cars'],
        'Psychological/Thriller': ['Psychological', 'Thriller', 'Mystery', 'Dementia'],
        'Other/Uncategorized': ['Kids', 'Hentai', 'Yaoi', 'Yuri', 'Ecchi', 'unknown']
    }
    df['genre'] = df['genre'].fillna('unknown').apply(
        lambda x: [genre.strip() for genre in x.split(',')] if isinstance(x, str) else []
    )
    categories_df = df['genre'].apply(lambda genres: map_genres_to_categories(genres, genre_categories))
    categories_df = pd.DataFrame(categories_df.tolist(), index=df.index)
    df = pd.concat([df, categories_df], axis=1)
    #df['genre'] = df['genre'].apply(lambda x: [genre.strip() for genre in x] if isinstance(x, str) else []) #Обработка пустых значений
    #categories_df = df['genre'].apply(map_genres_to_categories, genre_categories=genre_categories)
    #categories_df = pd.DataFrame(categories_df.tolist(), index=df.index)
    #df = pd.concat([df, categories_df], axis=1)
    return df


def one_hot_encode(df: pd.DataFrame, column: str) -> pd.DataFrame:
    df = pd.get_dummies(df, columns=[column], drop_first=False)
    return df


def clean_data(df: pd.DataFrame, df_users: pd.DataFrame) -> pd.DataFrame:
    df['genre'] = df['genre'].fillna('unknown')
    df['type'] = df['type'].fillna(df['type'].mode()[0])
    df['rating'] = df['rating'].fillna(df['rating'].mean())
    df['len_of_title'] = df['name'].apply(len)
    df = calculate_genre_stats(df)
    user_activity = calculate_user_activity(df_users)
    df['user_activity'] = df['anime_id'].map(user_activity)
    df = calculate_anime_type_features(df)
    df['user_activity'] = df['user_activity'].fillna(0)
    df['episodes'] = pd.to_numeric(df['episodes'], errors='coerce')
    df['episodes'] = df['episodes'].fillna(df['episodes'].mean())
    df = add_genre_categories(df)
    df['episode_density'] = df['members'] / df['episodes']  # Количество эпизодов в расчете на популярность (чем больше серий, тем меньше шанс, что они все посмотрят):

    # Ultra new super features - hopes die last...
    df['episodes_per_type'] = df['episodes'] / df['anime_count_by_type'] # Плотность серий(отслеживать уникальные типо)
    df['type_user_activity_ratio'] = df['user_activity'] / df['anime_count_by_type'] # активность по типу аниме
    # try more
    df['genre_count'] = df['genre'].apply(lambda x: len(x) if isinstance(x, list) else 0) # cчетчик жанров
    df['total_rated'] = df_users.groupby('anime_id').size() # количество людей оценивших впринципе
    df['total_rated'] = df['anime_id'].map(df['total_rated']).fillna(0)
    # количество людей поставивших максимальную оценку
    max_rating = df_users['rating'].max()
    df['favourite_count'] = df_users[df_users['rating'] == max_rating].groupby('anime_id').size()
    df['favourite_count'] = df['anime_id'].map(df['favourite_count']).fillna(0)
    # количество слов
    df['word_count'] = df['name'].str.split().str.len()

    # logirovanie
    df['log_episodes_per_type'] = np.log(df['episodes_per_type'] + 10 ** -6)
    df['log_members'] = np.log(df['members'] + 10 ** -6)  # Доавляем 1, чтобы избежать log(0)
    df['log_episodes'] = np.log(df['episodes'] + 10 ** -6)
    df['log_len_of_title'] = np.log(df['len_of_title'] + 10 ** -6)
    df['log_avg_members_genre'] = np.log(df['avg_members_genre'] + 10 ** -6)
    df['log_episode_density'] = np.log(df['episode_density'] + 10 ** -6)
    df['log_anime_count_by_type'] = np.log(df['anime_count_by_type'] + 10 ** -6)
    # df['log_avg_rating_by_type'] = np.log(df['avg_rating_by_type'] + 10 ** -6)
    df['log_user_activity'] = np.log(df['user_activity'] + 10 ** -6)
    df['log_genre_count'] = np.log(df['genre_count'] + 10 ** -6)
    df['log_total_rated'] = np.log(df['total_rated'] + 10 ** -6)
    df['log_favourite_count'] = np.log(df['favourite_count'] + 10 ** -6)
    df['log_word_count'] = np.log(df['word_count'] + 10 ** -6)
    df.drop(["episodes_per_type", "name","genre", "anime_id", "members", "avg_members_genre", "episodes", "len_of_title", "anime_count_by_type", "episode_density", 'user_activity', "genre_count", "total_rated", "favourite_count", "word_count"], axis=1, inplace=True)
    df = one_hot_encode(df, 'type')
    print("\nПропущенные значения после обработки:")
    print(df.isnull().sum())  # Вывод количества пропущенных значений после обработки
    return df


if __name__ == '__main__':
    args = parser_args_for_sac()
    try:
        with open(args.params, 'r') as f:
            params_all = yaml.safe_load(f)
            params = params_all['data_preparation']

        input_dir = Path(args.input_dir)
        output_dir = Path(args.output_dir)
        users_file_path = Path(args.users_file)
        output_dir.mkdir(exist_ok=True, parents=True)

        anime_data = pd.read_csv(list(input_dir.glob('*.csv'))[0]) # как работает?
        # эквивалентно ли - anime_data = pd.read_csv(input_dir)
        users_data = pd.read_csv(users_file_path)

        cleaned_data = clean_data(df=anime_data, df_users=users_data)

        X, y = cleaned_data.drop("rating", axis=1), cleaned_data['rating']

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
