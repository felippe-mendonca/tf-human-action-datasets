import argparse
from os.path import join
import json

import numpy as np
from xgboost import XGBClassifier
import pickle
from sklearn.metrics import accuracy_score

import tensorflow as tf
tf.enable_eager_execution()
tf.set_random_seed(1234)

from datasets.tfrecords.features import decode
from models.gesture_localization.encoding import DataEncoder
from models.options.options_pb2 import TrainXGBoostGestureLocalizationOptinons, Datasets
from models.options.utils import load_options, make_description
from utils.logger import Logger

tf.logging.set_verbosity(tf.logging.INFO)
log = Logger('XGBoost')


def load_metadata(dataset_folder, dataset_type=None):
    filename = join(dataset_folder, dataset_type + '.json')
    with open(filename, 'r') as f:
        data = json.load(f)
    gesture, not_gesture = 0, 0
    for _, value in data.items():
        gesture += value['samples']['gesture']
        not_gesture += value['samples']['not_gesture']
    return gesture, not_gesture


def main(options_filename, model_filename):
    op = load_options(options_filename, TrainXGBoostGestureLocalizationOptinons)
    dataset_name = Datasets.Name(op.dataset).lower()
    dataset_folder = join(op.storage.datasets_folder, dataset_name)
    dataset_tfrecords_folder = join(dataset_folder, '{}_tfrecords'.format(dataset_name))

    encoder = DataEncoder(dataset_folder=dataset_folder)
    mean_data, std_data = encoder.get_dataset_stats()
    mean_data, std_data = tf.constant(mean_data), tf.constant(std_data)

    def make_dataset(dataset_type):
        def standardize_feature(feature, label):
            feature = (feature - mean_data) / std_data
            return feature, label

        folder = join(dataset_tfrecords_folder, dataset_type)
        files = tf.data.Dataset().list_files(join(folder, 'Sample*.tfrecords'))
        dataset = tf.data.TFRecordDataset(filenames=files)
        dataset = dataset.map(decode)
        dataset = dataset.map(standardize_feature)
        return dataset

    def part_metadata(part):
        return load_metadata(dataset_tfrecords_folder, part)


    log.info("Loading model from '{}'", model_filename)
    xgboost = pickle.load(open(model_filename, "rb"))

    log.info("Loading train dataset")
    train_gesture, train_not_gesture = part_metadata('train')
    batch_size = train_gesture + train_not_gesture
    train_dataset = make_dataset('train').batch(batch_size=batch_size)

    X_train, y_train = train_dataset.make_one_shot_iterator().get_next()
    X_train, y_train = np.array(X_train), np.array(y_train)

    y_pred = xgboost.predict(X_train)
    y_pred = [round(value) for value in y_pred]

    acc = accuracy_score(y_train, y_pred)
    log.info("Train accuracy: {:.4f}", acc)

    del X_train, y_train, y_pred

    log.info("Loading validation dataset")
    validation_gesture, validation_not_gesture = part_metadata('validation')
    batch_size = validation_gesture + validation_not_gesture
    validation_dataset = make_dataset('validation').batch(batch_size=batch_size)

    X_validation, y_validation = validation_dataset.make_one_shot_iterator().get_next()
    X_validation, y_validation = np.array(X_validation), np.array(y_validation)

    y_pred = xgboost.predict(X_validation)
    y_pred = [round(value) for value in y_pred]

    acc = accuracy_score(y_validation, y_pred)
    log.info("Validation accuracy: {:.4f}", acc)

    del X_validation, y_validation, y_pred

    log.info("Loading test dataset")
    test_gesture, test_not_gesture = part_metadata('test')
    batch_size = test_gesture + test_not_gesture
    test_dataset = make_dataset('test').batch(batch_size=batch_size)

    X_test, y_test = test_dataset.make_one_shot_iterator().get_next()
    X_test, y_test = np.array(X_test), np.array(y_test)

    y_pred = xgboost.predict(X_test)
    y_pred = [round(value) for value in y_pred]

    acc = accuracy_score(y_test, y_pred)
    log.info("Test accuracy: {:.4f}", acc)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--options',
        required=True,
        type=str,
        help="""Path to options *.json file that matches with a 
        TrainXGBoostGestureLocalizationOptinons protobuf message.""")
    parser.add_argument(
        '--model',
        required=True,
        type=str,
        help="""Path to a *.pickle.dat file referred to a XGBoost 
        trained model.""")
    args = parser.parse_args()

    main(options_filename=args.options, model_filename=args.model)
