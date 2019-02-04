import re
import argparse
from os.path import join, exists, basename
from functools import reduce
import json

import numpy as np
import tensorflow as tf
tf.set_random_seed(1234)

from tensorflow.python.keras.optimizers import Adam, SGD
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger
from tensorflow.python.keras.models import load_model

from datasets.tfrecords.features import decode
from models.gesture_localization.model import make_model
from models.gesture_localization.encoding import DataEncoder
from models.options.options_pb2 import GestureLocalizationOptions, Datasets, Optimizers
from models.options.utils import load_options
from models.base.utils import get_logs_dir, gen_model_name
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


def load_metadata(dataset_folder, dataset_type=None):
    filename = join(dataset_folder, dataset_type + '.json')
    with open(filename, 'r') as f:
        data = json.load(f)
    gesture, not_gesture = 0, 0
    for _, value in data.items():
        gesture += value['samples']['gesture']
        not_gesture += value['samples']['not_gesture']
    return gesture, not_gesture


def main(options_filename, model_file=None, weights=None, reset_lr=False):
    op = load_options(options_filename, GestureLocalizationOptions)
    dataset_name = Datasets.Name(op.dataset).lower()
    dataset_folder = join(op.storage.datasets_folder, dataset_name)
    dataset_tfrecords_folder = join(dataset_folder, '{}_tfrecords'.format(dataset_name))

    encoder = DataEncoder(dataset_folder=dataset_folder)
    mean_data, std_data = encoder.get_dataset_stats()
    mean_data, std_data = tf.constant(mean_data), tf.constant(std_data)

    def make_dataset(dataset_type):
        def make_one_hot_label(feature, label):
            return feature, tf.one_hot(label, 2)

        def standardize_feature(feature, label):
            feature = (feature - mean_data) / std_data
            return feature, label

        folder = join(dataset_tfrecords_folder, dataset_type)
        files = tf.data.Dataset().list_files(join(folder, 'Sample*.tfrecords'))
        dataset = tf.data.TFRecordDataset(filenames=files)
        dataset = dataset.map(decode)
        dataset = dataset.map(standardize_feature)
        dataset = dataset.map(make_one_hot_label)
        return dataset

    train_dataset = make_dataset('train')
    validation_dataset = make_dataset('validation')
    test_dataset = make_dataset('test')

    def part_metadata(part):
        return load_metadata(dataset_tfrecords_folder, part)

    train_gesture, train_not_gesture = part_metadata('train')
    validation_gesture, validation_not_gesture = part_metadata('validation')
    test_gesture, test_not_gesture = part_metadata('test')
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

    if model_file is None:
        model_name = gen_model_name()
        initial_epoch = 0
        model = make_model(
            182, hidden_layers=op.hidden_layers, print_summary=True, name=model_name)
        if op.optimizer.type == Optimizers.Value('NOT_SPECIFIED'):
            optimizer = 'sgd'
        else:
            optimizer = Optimizers.Name(op.optimizer.type).lower()
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        if weights is not None:
            model.load_weights(filepath=weights, by_name=True)
    else:
        log.info("Restoring model from '{}'.", model_file)
        model = load_model(filepath=model_file, compile=True)
        model_name = model.name
        mf_match = re.match('^model-([0-9]{4,})-[0-9]{1}.[0-9]{4}.hdf5$', basename(model_file))
        initial_epoch = 0 if mf_match is None else int(mf_match.groups()[0])

    logs_dir = get_logs_dir(options=op, model_name=model_name)
    ckpt_log_dir = join(logs_dir, 'model-{epoch:04d}-{val_acc:.4f}.hdf5')
    csv_log_dir = join(logs_dir, 'logs.csv')

    callbacks = [ModelCheckpoint(filepath=ckpt_log_dir)]

    if op.optimizer.type == Optimizers.Value('NOT_SPECIFIED'):
        lr = op.optimizer.learning_rate
        decay = op.optimizer.learning_decay
        if model_file is not None and reset_lr:
            lr_scheduler = lambda epoch, _: lr * np.exp(-decay * (epoch - initial_epoch))
        else:
            lr_scheduler = lambda epoch, _: lr * np.exp(-decay * epoch)
        callbacks += [LearningRateScheduler(schedule=lr_scheduler)]

    callbacks += [
        TensorBoardMetrics(log_dir=logs_dir),
        TelegramExporter(telegram_id=op.telegram.id, token=op.telegram.token),
        CSVLogger(filename=csv_log_dir)
    ]

    model.fit(
        x=train_dataset,
        epochs=op.training.num_epochs,
        initial_epoch=initial_epoch,
        steps_per_epoch=steps_per_epoch,
        validation_data=validation_dataset,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--options', required=True, type=str, help='Path to options .json file')
    parser.add_argument(
        '--model-file',
        type=str,
        help='Path to .hdf5 model file to pre-load model and their weights.')
    parser.add_argument(
        '--weights', type=str, help='Path to .hdf5 model file containing model weights.')
    parser.add_argument('--reset-lr', action='store_true')
    args = parser.parse_args()
    main(
        options_filename=args.options,
        model_file=args.model_file,
        weights=args.weights,
        reset_lr=args.reset_lr)
