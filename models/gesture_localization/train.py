import argparse
from os.path import join, exists
from functools import reduce
import json

import numpy as np
import tensorflow as tf
tf.set_random_seed(1234)

from tensorflow.python.keras.optimizers import Adam, SGD
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger

from datasets.tfrecords.features import decode
from models.gesture_localization.model import make_model, EvalTrainDataset
from models.options.options_pb2 import GestureLocalizationOptions, Datasets
from models.options.utils import load_options
from models.base.callbacks import TensorBoardMetrics, LearningRateScheduler, TelegramExporter
from utils.logger import Logger
"""
Monkey patch to fix issue on 'standardize_single_array' function.
GitHub issue: https://github.com/tensorflow/tensorflow/issues/24520
GitHub PR: https://github.com/tensorflow/tensorflow/pull/24522

IMPORTANT: this patch must be placed after all tensorflow imports to 
           prevent function redefinitions, invalidating this patch.
"""
from tensorflow.python.keras import engine
from tf_patch.training_utils import standardize_single_array as _standardize_single_array
engine.training_utils.standardize_single_array = _standardize_single_array

tf.logging.set_verbosity(tf.logging.INFO)

log = Logger('GestureLocalizationTrain')


def load_metadata(dataset_folder, dataset_type):
    filename = join(dataset_folder, dataset_type + '.json')
    with open(filename, 'r') as f:
        data = json.load(f)
    gesture, not_gesture = 0, 0
    for _, value in data.items():
        gesture += value['samples']['gesture']
        not_gesture += value['samples']['not_gesture']
    return gesture, not_gesture


def main(options_filename):
    op = load_options(options_filename, GestureLocalizationOptions)
    dataset_name = Datasets.Name(op.dataset).lower()
    dataset_folder = join(op.storage.datasets_folder, dataset_name,
                          '{}_tfrecords'.format(dataset_name))

    def make_dataset(dataset_type):
        def make_one_hot_label(feature, label):
            return feature, tf.one_hot(label, 2)

        folder = join(dataset_folder, dataset_type)
        files = tf.data.Dataset().list_files(join(folder, 'Sample*.tfrecords'))
        dataset = tf.data.TFRecordDataset(filenames=files)
        dataset = dataset.map(decode)
        dataset = dataset.map(make_one_hot_label)
        return dataset

    train_dataset = make_dataset('train')
    validation_dataset = make_dataset('validation')
    test_dataset = make_dataset('test')

    train_gesture, train_not_gesture = load_metadata(dataset_folder, 'train')
    validation_gesture, validation_not_gesture = load_metadata(dataset_folder, 'validation')
    test_gesture, test_not_gesture = load_metadata(dataset_folder, 'test')
    steps_per_epoch = int((train_gesture + train_not_gesture) / op.training.batch_size)
    validation_steps = int((validation_gesture + validation_not_gesture) / op.training.batch_size)

    train_dataset = train_dataset                                      \
        .repeat()                                                      \
        .shuffle(buffer_size=op.training.shuffle_size)                 \
        .batch(batch_size=op.training.batch_size, drop_remainder=True) \
        .prefetch(buffer_size=op.training.prefetch_size)
    validation_dataset = validation_dataset                            \
        .repeat()                                                      \
        .batch(batch_size=op.training.batch_size, drop_remainder=True) \
        .prefetch(buffer_size=op.training.prefetch_size)

    model = make_model(182, hidden_layers=op.hidden_layers, print_summary=True)
    model.compile(
        optimizer=SGD(),
        loss='binary_crossentropy',
        metrics=['accuracy'])

    ckpt_log_dir = join(op.storage.logs, 'model-{epoch:04d}-{val_acc:.2f}.hdf5')
    csv_log_dir = join(op.storage.logs, 'logs.csv')
    model.fit(
        x=train_dataset,
        epochs=op.training.num_epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=validation_dataset,
        validation_steps=validation_steps,
        callbacks=[
            ModelCheckpoint(filepath=ckpt_log_dir),
            LearningRateScheduler(
                lambda epoch, _: op.optimizer.learning_rate * np.exp(-op.optimizer.learning_decay * epoch)
            ),
            TensorBoardMetrics(log_dir=op.storage.logs, model=model),
            TelegramExporter(telegram_id=op.telegram.id, token=op.telegram.token),
            CSVLogger(filename=csv_log_dir)
        ],
        verbose=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--options', required=True, type=str, help='Path to options .json file')

    args = parser.parse_args()
    main(options_filename=args.options)
