import argparse
import tensorflow as tf
from tensorflow import keras
import yaml
from tensorflow.keras.layers import Dense
import datetime
import shutil
from sklearn import preprocessing
from tensorflow.keras import Model
from pathlib import Path
import pandas as pd

BATCH_SIZE = 64   
BUFFER_SIZE = 1000   

def parser_args_for_sac():
    parser = argparse.ArgumentParser(description='Paths parser')
    parser.add_argument('--input_dir', '-id', type=str, default='data/prepared/',
                        required=False, help='path to input data directory')
    parser.add_argument('--output_dir', '-od', type=str, default='data/models/',
                        required=False, help='path to save prepared data')
    parser.add_argument('--baseline_model', '-bm', type=str, default='data/models/LinearRegression_prod.joblib',
                        required=False, help='path to linear regression prod version')
    parser.add_argument('--model_name', '-mn', type=str, default='LR', required=False,
                        help='file with dvc stage params')
    parser.add_argument('--params', '-p', type=str, default='params.yaml', required=False,
                        help='file with dvc stage params')
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

@tf.keras.utils.register_keras_serializable() #  Декоратор позволяет сериализовать и десериализовать модель для сохранения и загрузки.
class SomeModel(Model):
    def __init__(self, neurons_cnt=64, **kwargs):
        super(SomeModel, self).__init__(**kwargs)
        self.neurons_cnt = neurons_cnt  # Сохраняем значение параметра для конфигурации
        self.d_in = Dense(27, activation='relu')
        self.d1 = Dense(neurons_cnt, activation='relu')
        self.d2 = Dense(neurons_cnt, activation='relu')
        self.d3 = Dense(neurons_cnt, activation='relu')
        self.d_out = Dense(1)

    def call(self, x):
        x = self.d_in(x)
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        return self.d_out(x)
         
    def build(self, input_shape): # надо явно определить для построения
        super(SomeModel, self).build(input_shape)
        
    def get_config(self): 
        # Возвращаем параметры модели, включая кастомные
        config = super(SomeModel, self).get_config()
        config.update({
            "neurons_cnt": self.neurons_cnt  # Добавляем кастомный параметр в конфигурацию
        })
        return config

    @classmethod
    def from_config(cls, config):
        # Создаём экземпляр класса из конфигурации
        return cls(**config)

if __name__=="__main__":

    args = parser_args_for_sac()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    with open(args.params, 'r') as f:
        params_all = yaml.safe_load(f)

    output_dir.mkdir(exist_ok=True, parents=True)
    output_model_path = output_dir / (args.model_name + '.csv')
    output_model_joblib_path = output_dir / (args.model_name + '.joblib')
    

    input_dir = Path('./data/prepared')
    logs_path = Path('./data/logs') 

    if logs_path.exists():
        shutil.rmtree(logs_path) # удаляем, если существует /logs
    logs_path.mkdir(parents=True)
    params = params_all['neuralnet']

    X_train_name = input_dir / 'X_train.csv'
    y_train_name = input_dir / 'y_train.csv'
    X_test_name = input_dir / 'X_test.csv'
    y_test_name = input_dir / 'y_test.csv'

    X_train = pd.read_csv(X_train_name)
    y_train = pd.read_csv(y_train_name)
    X_test = pd.read_csv(X_test_name)
    y_test = pd.read_csv(y_test_name)

    X_train_np = X_train.to_numpy()
    y_train_np = y_train.to_numpy()

    y_train = pd.read_csv(y_train_name)
    y_test = pd.read_csv(y_test_name)

    train_ds = tf.data.Dataset.from_tensor_slices(
        (X_train, y_train)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(BATCH_SIZE)

    model = SomeModel(neurons_cnt=params.get("neurons")) # 64
    model.build(input_shape=(None, params.get("inputs")))  # 8

    loss_object = tf.keras.losses.MeanSquaredError() # что? 
    optimizer = tf.keras.optimizers.Adam(params.get("learning_rate"))

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.MeanAbsoluteError(name='train_mae')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.MeanAbsoluteError(name='test_mae')

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = logs_path / 'gradient_tape' / current_time / 'train'
    train_log_dir.mkdir(exist_ok=True, parents=True)
    test_log_dir = logs_path / 'gradient_tape' / current_time / 'test'
    test_log_dir.mkdir(exist_ok=True, parents=True)
    train_summary_writer = tf.summary.create_file_writer(str(train_log_dir))
    test_summary_writer = tf.summary.create_file_writer(str(test_log_dir))

    logdir=logs_path / "fit" / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir.mkdir(exist_ok=True, parents=True)
    fit_summary_writer = tf.summary.create_file_writer(str(logdir))

    tf.summary.trace_on(graph=True, profiler=True, profiler_outdir=str(logdir))

    for epoch in range(params.get("epochs")):
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
        print (template.format(epoch+1,
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
        
    model.save('./data/models/mymodel.keras')
    loaded_model = keras.models.load_model('./data/models/mymodel.keras')