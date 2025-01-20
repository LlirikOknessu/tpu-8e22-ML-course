import argparse
import yaml
import pandas as pd
import numpy as np
import shutil
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
from tensorflow import keras

def parser_args_for_sac():
    parser = argparse.ArgumentParser(description='Paths parser')
    parser.add_argument('--input_dir', '-id', type=str, default='./data/prepared/',
                        required=False, help='path to input data directory')
    parser.add_argument('--output_dir', '-od', type=str, default='./data/models/',
                        required=False, help='path to save prepared data')
    parser.add_argument('--params', '-p', type=str, default='params.yaml', required=False,
                        help='file with dvc stage params')
    parser.add_argument('--model_name', '-mn', type=str, required=False,
                        help='name of the model being trained')
    return parser.parse_args()


def load_data(input_dir):
    X_train = pd.read_csv(Path(input_dir) / 'X_train.csv')
    y_train = pd.read_csv(Path(input_dir) / 'y_train.csv').squeeze()
    X_test = pd.read_csv(Path(input_dir) / 'X_test.csv')
    y_test = pd.read_csv(Path(input_dir) / 'y_test.csv').squeeze()
    return X_train, y_train, X_test, y_test

def preprocess_data(X_train, X_test):
    one_hot_columns = ['type_Movie', 'type_Music', 'type_ONA', 'type_OVA', 'type_Special', 'type_TV']
    # One-hot decoding
    X_train['type_encoded'] = X_train[one_hot_columns].idxmax(axis=1).str.split('_').str[1]
    X_test['type_encoded'] = X_test[one_hot_columns].idxmax(axis=1).str.split('_').str[1]
    label_mapping = {label: idx for idx, label in enumerate(sorted(X_train['type_encoded'].unique()))}
    X_train['type_encoded'] = X_train['type_encoded'].map(label_mapping)
    X_test['type_encoded'] = X_test['type_encoded'].map(label_mapping)

    # Drop one-hot columns
    X_train = X_train.drop(columns=one_hot_columns)
    X_test = X_test.drop(columns=one_hot_columns)

    return X_train, X_test

@tf.function
def train_step(input_vector, labels):
    with tf.GradientTape() as tape:
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = model(input_vector, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)

@tf.function
def test_step(input_vector, labels):
    # training=False is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(input_vector, training=False)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)


def build_model(input_dim, neurons_cnt=64):
    model = models.Sequential()
    model.add(layers.Dense(27, activation='relu', input_dim=input_dim))
    model.add(layers.Dense(neurons_cnt, activation='relu'))
    model.add(layers.Dense(neurons_cnt, activation='relu'))
    model.add(layers.Dense(neurons_cnt, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss='mse',
                  metrics=['mae'])
    return model

if __name__ == '__main__':
    args = parser_args_for_sac()
    with open(args.params, 'r') as f:
        params_all = yaml.safe_load(f)
    params = params_all['NN']

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    logs_path = Path('./data/logs')

    BATCH_SIZE = params['BATCH_SIZE']  # 64
    BUFFER_SIZE = params['BUFFER_SIZE']  # 512
    LEARNING_RATE = params['LEARNING_RATE'] # 0.001
    EPOCHS = params['EPOCHS']  # 2000

    # Проверка наличия директорий
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory {input_dir} does not exist.")
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    if logs_path.exists():
        shutil.rmtree(logs_path)  # Удаляем, если существует /logs
    logs_path.mkdir(parents=True, exist_ok=True)

    loss_object = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    # Инициализация метрик
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.MeanAbsoluteError(name='train_mae')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.MeanAbsoluteError(name='test_mae')

    X_train, y_train, X_test, y_test = load_data(input_dir)
    X_train, X_test = preprocess_data(X_train, X_test)

    train_ds = tf.data.Dataset.from_tensor_slices(
        (X_train.to_numpy(), y_train.to_numpy())).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    test_ds = tf.data.Dataset.from_tensor_slices((X_test.to_numpy(), y_test.to_numpy())).batch(BATCH_SIZE)

    model = build_model(input_dim=X_train.shape[1], neurons_cnt=32)

    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = logs_path / 'gradient_tape' / current_time / 'train'
    train_log_dir.mkdir(parents=True, exist_ok=True)
    test_log_dir = logs_path / 'gradient_tape' / current_time / 'test'
    test_log_dir.mkdir(parents=True, exist_ok=True)
    train_summary_writer = tf.summary.create_file_writer(str(train_log_dir))
    test_summary_writer = tf.summary.create_file_writer(str(test_log_dir))

    logdir = logs_path / "fit" / current_time
    logdir.mkdir(exist_ok=True, parents=True)
    fit_summary_writer = tf.summary.create_file_writer(str(logdir))

    tf.summary.trace_on(graph=True, profiler=True, profiler_outdir=str(logdir))

    for epoch in range(EPOCHS):
        # Reset the metrics at the start of the next epoch
        for (x_train, y_train) in train_ds:
            with fit_summary_writer.as_default():
                train_step(x_train, y_train)

        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)

        for (x_test, y_test) in test_ds:
            test_step(x_test, y_test)

        with test_summary_writer.as_default():
            tf.summary.scalar('loss', test_loss.result(), step=epoch)
            tf.summary.scalar('mae', test_accuracy.result(), step=epoch)

        template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test MAE: {}'
        print(template.format(epoch + 1,
                              train_loss.result(),
                              train_accuracy.result(),
                              test_loss.result(),
                              test_accuracy.result()))

        # Reset metrics every epoch
        train_loss.reset_state()
        test_loss.reset_state()
        train_accuracy.reset_state()
        test_accuracy.reset_state()

    with fit_summary_writer.as_default():
        tf.summary.trace_export(
            name="my_func_trace",
            step=0,
            profiler_outdir=str(logdir)
        )

    # Save the model
    model.save(output_dir / f'nn_model.h5')
    print(f"Модель сохранена в data/models/nn_model.h5")