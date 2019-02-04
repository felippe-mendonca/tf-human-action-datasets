import re
import sys
import json
import argparse
from os import makedirs
from os.path import join, exists
from shutil import rmtree
import numpy as np
import pandas as pd
import tensorflow as tf

from utils.logger import Logger
from datasets.tfrecords.features import bytes_feature, int64_feature
from datasets.montalbanov2.base import Reader, PoseIterator
from models.gesture_localization.encoding import DataEncoder
from models.base.utils import StatsRecorder

log = Logger(name='montalbanov2_create_tfrecords')


def main(dataset_folder, output_folder, parts):

    encoder = DataEncoder(dataset_folder)

    def make_tfrecords(dataset_part):
        tfrecords_folder = join(output_folder, 'montalbanov2_tfrecords')
        part_folder = join(tfrecords_folder, dataset_part)
        if exists(part_folder):
            rmtree(part_folder)
        makedirs(part_folder)

        stats_recorder = StatsRecorder()
        reader = Reader(folder=join(dataset_folder, dataset_part))
        all_metadata = {}
        for sample, poses, labels in reader:

            log.info('[Start] {}', sample)
            tfrecord_filename = join(part_folder, sample + '.tfrecords')
            writer = tf.python_io.TFRecordWriter(tfrecord_filename)

            one_hot_labels = np.zeros(poses.shape[0])
            for _, label in labels.iterrows():
                one_hot_labels[label['begin']:label['end'] + 1] = 1

            metadata = {'samples': {'gesture': 0, 'not_gesture': 0}}
            encoder.reset()
            sample_features = []
            for pose, label in zip(PoseIterator(poses), one_hot_labels):
                if np.all(pose == 0.0):
                    continue

                vec_features = encoder.encode(pose)
                if vec_features is None:
                    continue

                shape = np.array(vec_features.shape, dtype=np.int32)
                feature = {
                    'label': int64_feature(int(label)),
                    'positions': bytes_feature(vec_features.tobytes()),
                    'shape': bytes_feature(shape.tobytes())
                }

                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())
                if int(label) == 1:
                    metadata['samples']['gesture'] += 1
                else:
                    metadata['samples']['not_gesture'] += 1

                if dataset_part == 'train':
                    sample_features.append(vec_features)

            writer.close()
            sys.stdout.flush()
            all_metadata[sample] = metadata
            if dataset_part == 'train' and len(sample_features) > 0:
                stats_recorder.update(np.vstack(sample_features))

            log.info('[Done] {}', sample)

        with open(part_folder + '.json', 'w') as f:
            json.dump(obj=all_metadata, fp=f, sort_keys=True, indent=2)

        if dataset_part == 'train':
            filename = join(tfrecords_folder, 'mean_features.csv')
            df = pd.DataFrame(data=np.expand_dims(stats_recorder.mean, axis=0))
            df.to_csv(path_or_buf=filename, index=False, header=False)

            filename = join(tfrecords_folder, 'std_features.csv')
            df = pd.DataFrame(data=np.expand_dims(stats_recorder.std, axis=0))
            df.to_csv(path_or_buf=filename, index=False, header=False)

    available_parts = ['train', 'validation', 'test']
    for part in parts:
        if part not in available_parts:
            log.warn("Skipping part '{}'. Only {} are available.", part,
                     ', '.join(available_parts))
            continue

        make_tfrecords(part)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        required=True,
        type=str,
        help='Directory containing MontalbanoV2 dataset with *.csv files.')
    parser.add_argument(
        '--output',
        required=False,
        type=str,
        default='.',
        help='Directory to save .tfrecord files.')
    parser.add_argument(
        '--dataset-parts',
        type=str,
        nargs='+',
        default=['train', 'validation', 'test'],
        help=
        'Dataset parts that will have tfrecords file generated. Specify a list with train, validation and/or test.'
    )

    args = parser.parse_args()
    main(dataset_folder=args.dataset, output_folder=args.output, parts=args.dataset_parts)