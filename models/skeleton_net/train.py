import argparse
import logging

import tensorflow as tf
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.estimator import model_to_estimator

from skeletons_datasets.common.reader import DatasetReader
from skeletons_datasets.ntu_rgbd.base import ONE_PERSON_ACTION
from models.skeleton_net.encoding import DataEncoder
from models.skeleton_net.model import make_model
from models.options.options_pb2 import SkeletonNetOptions, Datasets
from models.options.utils import load_options

tf.logging.set_verbosity(logging.INFO)

def main(options_filename):
    op = load_options(options_filename, SkeletonNetOptions)
    tfrecord_basename = '{}.{{}}.tfrecords'.format(Datasets.Name(op.dataset).lower())

    # add shuffle option
    train_dataset = DatasetReader(
        filenames=tfrecord_basename.format('train'),
        batch_size=op.training.batch_size,
        drop_reminder=True,
        num_epochs=op.training.num_epochs,
        shuffle_size=op.training.shuffle_size,
        prefetch_size=op.training.prefetch_size)
    test_dataset = DatasetReader(
        filenames=tfrecord_basename.format('test'),
        batch_size=1,
        drop_reminder=True,
        num_epochs=1,
        shuffle_size=None,
        prefetch_size=None)

    encoder = DataEncoder(
        output_shape=[op.input_shape.width, op.input_shape.height],
        one_hot=True,
        n_classes=len(ONE_PERSON_ACTION),
        label_offset_to_zero=1)

    encoder.apply_to_dataset(train_dataset)
    encoder.apply_to_dataset(test_dataset)

    n_classes = len(ONE_PERSON_ACTION)
    model = make_model(
        input_shape=(op.input_shape.width, op.input_shape.height, 3),
        n_parallel_paths=5,
        n_classes=n_classes)
    model.compile(
        optimizer=Adam(lr=op.optimizer.learning_rate, decay=op.optimizer.learning_decay),
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    config = tf.estimator.RunConfig(
        save_summary_steps=op.estimator.save_summary_steps,
        save_checkpoints_secs=op.estimator.save_checkpoints_secs,
        keep_checkpoint_max=op.estimator.keep_checkpoint_max,
        log_step_count_steps=op.estimator.log_step_count_steps)

    # verify if 'model_dir' parameter must be a full path directory
    estimator = model_to_estimator(keras_model=model, model_dir=op.storage.logs, config=config)

    body_parts = sorted(encoder.get_body_parts().keys())

    def make_input_fn(dataset, body_parts):
        def make_inputs(batch_labels, batch_features):
            features_dict = {
                'path{}/vgg16_input'.format(fid): batch_features[part]
                for fid, part in enumerate(body_parts)
            }
            return features_dict, batch_labels

        dataset.map(make_inputs)
        batch_features, batch_labels = dataset.get_inputs()
        return batch_features, batch_labels

    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda: make_input_fn(train_dataset, body_parts),
        max_steps=op.training.max_steps if op.training.max_steps > 0 else None)
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda: make_input_fn(test_dataset, body_parts))

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--options', required=True, type=str, help='Path to options .json file')

    args = parser.parse_args()
    main(options_filename=args.options)