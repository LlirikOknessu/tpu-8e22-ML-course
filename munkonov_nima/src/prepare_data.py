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

    # Новые признаки:
    # 1. Фактор количества учеников в классе. Если > 23 0 иначе 1
    scores['f_student_count'] = (scores['n_student'] < 23).map(int)

    # 2. Комплексный фактор школы и наличия субсидий
    comp = pd.crosstab([scores.school_type, scores.teaching_method],
                       [scores.lunch, scores.school_setting],
                       margins=True,
                       values=scores.posttest,
                       aggfunc='mean')
    comp.stack(level=['lunch', 'school_setting'], future_stack=True)
    comp.sort_values()
    scores['f_complex'] = 0.0
    indecies = ['school_type', 'teaching_method', 'lunch', 'school_setting']
    for i in range(scores.shape[0]):
        scores.at[i, 'f_complex'] = comp[tuple(scores.loc[i, indecies])]

    # 3. Средний балл класса за предварительный тест
    scores['f_class_mean'] = scores[['classroom',
                                     'pretest']].groupby(['classroom'
                                                          ]).transform('mean')

    # Экспортируем данные для обучения модели линейной регрессии
    scores = scores.select_dtypes(include=['number'])
    scores.to_csv(args.output_path, index=False)