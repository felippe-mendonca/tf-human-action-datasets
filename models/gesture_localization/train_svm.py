import argparse
from os.path import join
import json

import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.externals import joblib

import tensorflow as tf
tf.enable_eager_execution()
tf.set_random_seed(1234)

from datasets.tfrecords.features import decode
from models.gesture_localization.encoding import DataEncoder
from models.options.options_pb2 import GestureLocalizationOptions, Datasets
from models.options.utils import load_options, make_description
from utils.logger import Logger

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


def main(options_filename):
    op = load_options(options_filename, GestureLocalizationOptions)
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

    log.info("Loading training dataset")
    train_gesture, train_not_gesture = part_metadata('train')
    batch_size = train_gesture + train_not_gesture
    train_dataset = make_dataset('train').batch(batch_size=batch_size)

    X_train, y_train = train_dataset.make_one_shot_iterator().get_next()
    X_train, y_train = np.array(X_train), np.array(y_train)

    log.info("Loading validation dataset")
    validation_gesture, validation_not_gesture = part_metadata('validation')
    batch_size = validation_gesture + validation_not_gesture
    validation_dataset = make_dataset('validation').batch(batch_size=batch_size)

    X_validation, y_validation = validation_dataset.make_one_shot_iterator().get_next()
    X_validation, y_validation = np.array(X_validation), np.array(y_validation)

    n_estimators = 10
    kernels = ['linear', 'rbf', 'poly']
    for kernel in kernels:
        log.info('Starting training with kernel {}', kernel)
        clf = OneVsRestClassifier(
            BaggingClassifier(
                base_estimator=SVC(
                    kernel=kernel, class_weight='balanced', random_state=1234, verbose=True),
                max_samples=1.0 / n_estimators,
                n_estimators=n_estimators,
                random_state=1234,
                n_jobs=-1,
                verbose=1),
            n_jobs=-1)
        clf.fit(X_train, y_train)

        validation_score = clf.score(X_validation, y_validation)
        log.info("Validation accuracy: {:.4f}", validation_score)

        log.info("Saving model")
        filename = "svm_{}.sav".format(kernel)
        joblib.dump(value=clf, filename=filename, compress=('zlib', 9))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--options',
        required=True,
        type=str,
        help="""Path to options *.json file that matches with a 
        GestureLocalizationOptions protobuf message.""")
    args = parser.parse_args()

    main(options_filename=args.options)
