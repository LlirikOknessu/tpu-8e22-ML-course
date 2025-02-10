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

def load_data(input_dir):
    X_full = pd.read_csv(Path(input_dir) / 'X_full.csv')
    y_full = pd.read_csv(Path(input_dir) / 'y_full.csv').squeeze()
    return X_full, y_full

def build_model(input_dim, neurons_cnt=64):
    model = models.Sequential()
    model.add(layers.Dense(30, activation='relu', input_dim=input_dim))
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
    params = params_all['Neural_network']

    BATCH_SIZE = params['BATCH_SIZE']
    BUFFER_SIZE = params['BUFFER_SIZE']
    LEARNING_RATE = params['LEARNING_RATE']
    EPOCHS = params['EPOCHS']

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    logs_path = Path('./data/logs_prod')

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory {input_dir} does not exist.")
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    if logs_path.exists():
        shutil.rmtree(logs_path)
    logs_path.mkdir(parents=True, exist_ok=True)

    loss_object = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    # Инициализация метрик
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.MeanAbsoluteError(name='train_mae')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.MeanAbsoluteError(name='test_mae')

    X, y = load_data(input_dir)

    train_ds = tf.data.Dataset.from_tensor_slices(
        (X.to_numpy(), y.to_numpy())).batch(BATCH_SIZE)

    model = build_model(input_dim=X.shape[1])

    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = logs_path / 'gradient_tape' / current_time / 'train'
    train_log_dir.mkdir(parents=True, exist_ok=True)
    train_summary_writer = tf.summary.create_file_writer(str(train_log_dir))

    logdir = logs_path / "fit" / current_time
    logdir.mkdir(exist_ok=True, parents=True)
    fit_summary_writer = tf.summary.create_file_writer(str(logdir))

    tf.summary.trace_on(graph=True, profiler=True, profiler_outdir=str(logdir))

    for epoch in range(EPOCHS):
        for (x, y) in train_ds:
            with fit_summary_writer.as_default():
                train_step(x, y)

        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)

        template = 'Epoch {}, Loss: {}, Accuracy: {},'
        print(template.format(epoch + 1,
                              train_loss.result(),
                              train_accuracy.result()))

        train_loss.reset_state()
        train_accuracy.reset_state()

    with fit_summary_writer.as_default():
        tf.summary.trace_export(
            name="my_func_trace",
            step=0,
            profiler_outdir=str(logdir)
        )

    # Save the model
    model.save(output_dir / 'nn_model_prod.h5')
    print(f"Модель сохранена в data/models/nn_model_prod.h5")

    # predictions = model.predict(X.to_numpy()).flatten()
    # y_true = y.numpy().flatten()
    #
    # # Создание DataFrame
    # predictions_df = pd.DataFrame({
    #     'Prediction': predictions,
    #     'True_Value': y_true
    # })
    #
    # predictions_df.to_csv(output_dir / 'predictions.csv', index=False)
    # print(f"Предсказания сохранены в {output_dir / 'predictions.csv'}")

