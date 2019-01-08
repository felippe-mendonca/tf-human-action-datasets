import os
import argparse

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.estimator import model_to_estimator
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard

from datasets.tfrecords.features import decode
from datasets.ntu_rgbd.base import ONE_PERSON_ACTION, ACTION_NAMES
from models.skeleton_net.encoding import DataEncoder
from models.skeleton_net.model import make_model, InputsExporter
from models.options.options_pb2 import SkeletonNetOptions, Datasets
from models.options.utils import load_options

tf.logging.set_verbosity(tf.logging.INFO)


def main(options_filename):
    op = load_options(options_filename, SkeletonNetOptions)
    ds_basename = '{}.{{}}.tfrecords'.format(Datasets.Name(op.dataset).lower())

    train_n_samples = sum(1 for _ in tf.io.tf_record_iterator(path=ds_basename.format('train')))
    train_dataset = tf.data.TFRecordDataset(filenames=ds_basename.format('train'))
    test_dataset = tf.data.TFRecordDataset(filenames=ds_basename.format('test'))

    train_dataset = train_dataset.map(decode)
    test_dataset = test_dataset.map(decode)

    encoder = DataEncoder(
        output_shape=[op.input_shape.width, op.input_shape.height],
        one_hot=True,
        n_classes=len(ONE_PERSON_ACTION),
        label_offset_to_zero=1)

    train_dataset = encoder.apply_to_dataset(train_dataset)
    test_dataset = encoder.apply_to_dataset(test_dataset)

    body_parts = sorted(encoder.get_body_parts().keys())

    def make_inputs(batch_features, batch_labels):
        features_dict = {
            'path{}/vgg16_input'.format(fid): batch_features[part]
            for fid, part in enumerate(body_parts)
        }
        return features_dict, batch_labels

    train_dataset = train_dataset.map(make_inputs)
    test_dataset = test_dataset.map(make_inputs)

    train_dataset = train_dataset                                      \
        .repeat()                                                      \
        .shuffle(buffer_size=op.training.shuffle_size)                 \
        .batch(batch_size=op.training.batch_size, drop_remainder=True) \
        .prefetch(buffer_size=op.training.prefetch_size)
    test_dataset = test_dataset.batch(batch_size=1)

    n_classes = len(ONE_PERSON_ACTION)
    model = make_model(
        input_shape=(op.input_shape.width, op.input_shape.height, 3),
        n_parallel_paths=5,
        n_classes=n_classes)
    model.compile(
        optimizer=Adam(lr=op.optimizer.learning_rate, decay=op.optimizer.learning_decay),
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    ckpt_log_dir = os.path.join(op.storage.logs, 'cp-{epoch:04d}.ckpt')
    train_features, train_labels = train_dataset.make_one_shot_iterator().get_next()

    model.fit(
        x=train_features,
        y=train_labels,
        epochs=op.training.num_epochs,
        steps_per_epoch=int(train_n_samples / op.training.batch_size),
        callbacks=[
            ModelCheckpoint(filepath=ckpt_log_dir),
            TensorBoard(log_dir=op.storage.logs, batch_size=op.training.batch_size),
            InputsExporter(
                features=train_features,
                labels=train_labels,
                log_dir=op.storage.logs,
                class_labels=ACTION_NAMES,
                period=50)
        ],
        verbose=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--options', required=True, type=str, help='Path to options .json file')

    args = parser.parse_args()
    main(options_filename=args.options)