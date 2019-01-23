import argparse
from os.path import join, exists
from functools import reduce
import json

import tensorflow as tf
# tf.enable_eager_execution()

from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard

from datasets.tfrecords.features import decode
from models.gesture_localization.model import make_model
from models.options.options_pb2 import GestureLocalizationOptions, Datasets
from models.options.utils import load_options
from utils.logger import Logger

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
        def make_label(feature, label):
            label = tf.reshape(label, shape=(1, ))
            label = tf.cast(label, tf.float32)
            return feature, label

        folder = join(dataset_folder, dataset_type)
        files = tf.data.Dataset().list_files(join(folder, 'Sample*.tfrecords'))
        dataset = tf.data.TFRecordDataset(filenames=files)
        dataset = dataset.map(decode)
        dataset = dataset.map(make_label)
        return dataset

    train_dataset = make_dataset('train')
    validation_dataset = make_dataset('validation')
    train_gesture, train_not_gesture = load_metadata(dataset_folder, 'train')
    log.info('[Train] gesture: {} | not_gesture: {}', train_gesture, train_not_gesture)
    # validation_gesture, validation_not_gesture = load_metadata(dataset_folder, 'validation')

    train_dataset = train_dataset                                      \
        .repeat()                                                      \
        .shuffle(buffer_size=op.training.shuffle_size)                 \
        .batch(batch_size=op.training.batch_size, drop_remainder=True) \
        .prefetch(buffer_size=op.training.prefetch_size)
    validation_dataset = validation_dataset.batch(batch_size=1)

    train_features, train_labels = train_dataset.make_one_shot_iterator().get_next()

    model = make_model(182, hidden_neurons=op.hidden_neurons)
    model.compile(
        optimizer=RMSprop(lr=op.optimizer.learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy'])

    ckpt_log_dir = join(op.storage.logs, 'cp-{epoch:04d}.ckpt')
    model.fit(
        x=train_features,
        y=train_labels,
        epochs=op.training.num_epochs,
        steps_per_epoch=int((train_gesture + train_not_gesture) / op.training.batch_size),
        callbacks=[
            ModelCheckpoint(filepath=ckpt_log_dir),
            TensorBoard(log_dir=op.storage.logs, batch_size=op.training.batch_size)
        ],
        verbose=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--options', required=True, type=str, help='Path to options .json file')

    args = parser.parse_args()
    main(options_filename=args.options)
