import tensorflow as tf
import pandas as pd
import argparse
import yaml
from pathlib import Path
from joblib import dump


def parser_args_for_sac():
    parser = argparse.ArgumentParser(description='Paths parser')
    parser.add_argument('--input_dir', '-id', type=str,
                        default='data/prepared/',
                        required=False,
                        help='path to input data directory')
    parser.add_argument('--output_dir', '-od', type=str,
                        default='data/models/',
                        required=False,
                        help='path to save model')
    parser.add_argument('--model_name', '-mn', type=str,
                        default='NN_prod',
                        required=False,
                        help='name for the model')
    parser.add_argument('--logs_path', '-lp', type=str,
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
    output_dir.mkdir(exist_ok=True, parents=True)
    output_model_joblib_path = output_dir / (args.model_name + '_prod.joblib')

    X_full_name = input_dir / 'X_full.csv'
    y_full_name = input_dir / 'y_full.csv'

    X_full = pd.read_csv(X_full_name)
    y_full = pd.read_csv(y_full_name)\

    train_ds = tf.data.Dataset.from_tensor_slices(
        (X_full, y_full)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(7, )),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model.fit(train_ds, epochs=100)
    reg = model
    reg.fit(X_full, y_full)

    dump(reg, output_model_joblib_path)
    print(f"Production модель сохранена в {output_model_joblib_path}")
