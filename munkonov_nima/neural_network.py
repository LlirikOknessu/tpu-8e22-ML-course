import tensorflow as tf
import pandas as pd
import numpy as np
import argparse
import yaml
import datetime
import shutil
from pathlib import Path
from joblib import dump
from sklearn.metrics import mean_absolute_error, r2_score


def parser_args_for_sac():
    parser = argparse.ArgumentParser(description='Paths parser')
    parser.add_argument('--input_dir', '-id',
                        type=str,
                        default='data/prepared/',
                        required=False,
                        help='path to input data directory')
    parser.add_argument('--output_dir', '-od',
                        type=str,
                        default='data/models/',
                        required=False,
                        help='path to save prepared data')
    parser.add_argument('--model_name', '-mn',
                        type=str,
                        default='LR',
                        required=False,
                        help='file with dvc stage params')
    parser.add_argument('--logs_path', '-lp',
                        type=str,
                        default='data/logs/',
                        required=False,
                        help='path to logs dir')
    parser.add_argument('--params', '-p',
                        type=str,
                        default='params.yaml',
                        required=False,
                        help='file with dvc stage params')
    return parser.parse_args()


if __name__ == '__main__':
    args = parser_args_for_sac()

    with open(args.params, 'r') as f:
        params_all = yaml.safe_load(f)
    params = params_all['neural_network']

    BATCH_SIZE = params['batch_size']
    BUFFER_SIZE = params['buffer_size']
    LEARNING_RATE = params['learning_rate']
    EPOCHS = params['epochs']

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    logs_path = Path(args.logs_path)
    if logs_path.exists():
        shutil.rmtree(logs_path)
    logs_path.mkdir(parents=True)

    output_dir.mkdir(exist_ok=True, parents=True)
    output_model_joblib_path = output_dir / (args.model_name + '.joblib')

    X_train_name = input_dir / 'X_train.csv'
    y_train_name = input_dir / 'y_train.csv'
    X_test_name = input_dir / 'X_test.csv'
    y_test_name = input_dir / 'y_test.csv'

    X_train = pd.read_csv(X_train_name)
    y_train = pd.read_csv(y_train_name)
    X_test = pd.read_csv(X_test_name)
    y_test = pd.read_csv(y_test_name)

    train_ds = tf.data.Dataset.from_tensor_slices(
        (X_train, y_train)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    test_ds = tf.data.Dataset.from_tensor_slices(
        (X_test, y_test)).batch(BATCH_SIZE)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(7, )),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    log_filename = str(datetime.date.today())
    logdir = logs_path / log_filename
    logdir.mkdir(exist_ok=True, parents=True)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=str(logdir),
        histogram_freq=1,  # Записывать гистограммы весов каждые 1 эпоху
        write_graph=True,  # Включить сохранение графа
        write_images=True  # Сохранение весов как изображений
    )
    model.fit(train_ds,
              validation_data=test_ds,
              epochs=100,
              callbacks=[tensorboard_callback])
    reg = model

    predicted_values = np.squeeze(reg.predict(X_test))

    print("NEURAL NETWORK")
    print("Model MAE:    ", mean_absolute_error(y_test, predicted_values))
    r2 = r2_score(y_test, predicted_values)
    print(f"R² Score:      {r2:.4f}")

    dump(reg, output_model_joblib_path)
