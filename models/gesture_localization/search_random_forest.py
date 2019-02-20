import argparse
from os.path import join
import json
from google.protobuf.json_format import MessageToDict

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib

import tensorflow as tf
tf.enable_eager_execution()
tf.set_random_seed(1234)

from datasets.tfrecords.features import decode
from models.gesture_localization.encoding import DataEncoder
from models.options.options_pb2 import GridSearchGestureLocalizationOptions, Datasets
from models.options.utils import load_options, make_description
from utils.logger import Logger

tf.logging.set_verbosity(tf.logging.INFO)
log = Logger('SearchRandomForest')


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


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
    op = load_options(options_filename, GridSearchGestureLocalizationOptions)
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

    param_grid = MessageToDict(op.param_grid, preserving_proto_field_name=True)
    if 'None' in param_grid['max_features']:
        param_grid['max_features'].remove('None')
        param_grid['max_features'].append(None)

    rf = RandomForestClassifier()
    gs = GridSearchCV(estimator=rf, param_grid=param_grid, refit=True, cv=3, n_jobs=-1, verbose=1)

    log.info("Loading training dataset")
    train_gesture, train_not_gesture = part_metadata('train')
    batch_size = train_gesture + train_not_gesture
    batch_size = 100
    train_dataset = make_dataset('train').batch(batch_size=batch_size)

    X_train, y_train = train_dataset.make_one_shot_iterator().get_next()
    X_train, y_train = np.array(X_train), np.array(y_train)

    log.info('Starting grid search')
    gs.fit(X_train, y_train)

    log.info('Best params')
    print(gs.best_params_)

    params_filename = join(op.storage.logs, "random_forest_best_params.json")
    with open(params_filename, 'w') as f:
        json.dump(gs.best_params_, f, indent=2, sort_keys=True)

    results_filename = join(op.storage.logs, "random_forest_results.json")
    with open(results_filename, 'w') as f:
        json.dump(gs.cv_results_, f, indent=2, cls=NumpyEncoder)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--options',
        required=True,
        type=str,
        help="""Path to options *.json file that matches with a 
        GridSearchGestureLocalizationOptions protobuf message.""")
    args = parser.parse_args()

    main(options_filename=args.options)
