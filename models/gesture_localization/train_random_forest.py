import argparse
from os.path import join
import json

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

import tensorflow as tf
tf.enable_eager_execution()
tf.set_random_seed(1234)

from datasets.tfrecords.features import decode
from models.gesture_localization.encoding import DataEncoder
from models.options.options_pb2 import TrainRandomForestGestureLocalizationOptinons, Datasets
from models.options.utils import load_options, make_description
from utils.logger import Logger

tf.logging.set_verbosity(tf.logging.INFO)
log = Logger('RandomForestTrain')


def load_metadata(dataset_folder, dataset_type=None):
    filename = join(dataset_folder, dataset_type + '.json')
    with open(filename, 'r') as f:
        data = json.load(f)
    gesture, not_gesture = 0, 0
    for _, value in data.items():
        gesture += value['samples']['gesture']
        not_gesture += value['samples']['not_gesture']
    return gesture, not_gesture


def main(options_filename):
    op = load_options(options_filename, TrainRandomForestGestureLocalizationOptinons)
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

    with open(op.params_file, 'r') as f:
        params = json.load(f)

    log.info("Loading training dataset")
    train_gesture, train_not_gesture = part_metadata('train')
    batch_size = train_gesture + train_not_gesture
    train_dataset = make_dataset('train').batch(batch_size=batch_size)

    X_train, y_train = train_dataset.make_one_shot_iterator().get_next()
    X_train, y_train = np.array(X_train), np.array(y_train)

    rf = RandomForestClassifier(**params, n_jobs=-1, random_state=1234)

    log.info("Starting training with parameters:\n{}", json.dumps(params, indent=2))
    rf.fit(X_train, y_train)

    log.info("Loading validation dataset")
    validation_gesture, validation_not_gesture = part_metadata('validation')
    batch_size = validation_gesture + validation_not_gesture
    validation_dataset = make_dataset('validation').batch(batch_size=batch_size)

    X_validation, y_validation = validation_dataset.make_one_shot_iterator().get_next()
    X_validation, y_validation = np.array(X_validation), np.array(y_validation)

    validation_score = rf.score(X_validation, y_validation)
    log.info("Validation accuracy: {:.4f}", validation_score)

    filename = join(op.storage.logs, "random_forest.sav")
    log.info("Saving model on '{}'", filename)
    joblib.dump(value=rf, filename=filename, compress=('zlib', 9))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--options',
        required=True,
        type=str,
        help="""Path to options *.json file that matches with a 
        TrainRandomForestGestureLocalizationOptinons protobuf message.""")
    args = parser.parse_args()

    main(options_filename=args.options)
