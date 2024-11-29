# Подготовка производимая в еде, только без мишуры
# Импортируем все необходимые библиотеки
import pandas as pd
from pathlib import Path


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', '-ip', type=Path, required=True)
    parser.add_argument('--output_path', '-op', type=Path, required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    # Считываем данные, выводим размер таблицы
    scores = pd.read_csv(args.input_path)

    # Исправим данные в столбце
    real_n_student = scores['classroom'].value_counts()
    for i in range(scores.shape[0]):
        scores.at[i, 'n_student'] = real_n_student[scores.loc[i, 'classroom']]
    
    scores['f_lunch'] = scores['lunch'].replace(['Qualifies for reduced/free lunch', 'Does not qualify'], [0, 1])
    scores['f_teaching_method'] = scores['teaching_method'].replace(['Standard', 'Experimental'], [0, 1])
    scores['f_school_setting'] = scores['school_setting'].replace(['Rural', 'Urban', 'Suburban'], [0, 1, 2])
    scores['f_school_type'] = scores['school_type'].replace(['Public', 'Non-public'], [0, 1])

    # Экспортируем данные для обучения модели линейной регрессии
    scores = scores.select_dtypes(include=['number'])
    scores.to_csv(args.output_path, index=False)