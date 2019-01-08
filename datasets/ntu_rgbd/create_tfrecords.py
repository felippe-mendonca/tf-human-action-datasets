import sys
import logging
import argparse
from os import makedirs
from os.path import join, exists
from shutil import rmtree
import numpy as np
import tensorflow as tf

from utils.logger import Logger
from datasets.tfrecords.features import bytes_feature, int64_feature
from datasets.ntu_rgbd.base import Loader

log = Logger(name='ntu_rgbd_create_tfrecords')


def main(dataset_folder, output_folder):
    loader = Loader(folder=dataset_folder, load_headings=False)

    def make_tfrecords(dataset_part):
        part_folder = join(output_folder, 'ntu_rgbd', dataset_part)
        if exists(part_folder):
            rmtree(part_folder)
        makedirs(part_folder)

        def tfrecord_filename(label):
            return join(part_folder, '{label}.tfrecords'.format(label=label))

        writers = {}
        files = loader.list_files(only_valids=True, dataset_part=dataset_part)

        for n, file in enumerate(files):
            metadata, data = loader.load_from_file(file)
            positions_list = list(data.values())[0]['positions']

            positions = np.dstack(positions_list)
            shape = np.array(positions.shape, dtype=np.int32)
            label = metadata['action']

            feature = {
                'label': int64_feature(label),
                'positions': bytes_feature(positions.tobytes()),
                'shape': bytes_feature(shape.tobytes())
            }

            example = tf.train.Example(features=tf.train.Features(feature=feature))

            if label not in writers:
                writers[label] = tf.python_io.TFRecordWriter(tfrecord_filename(label))

            writers[label].write(example.SerializeToString())

            if n % 100 == 0:
                log.info('{}/{}'.format(n, len(files)))

        for _, writer in writers.items():
            writer.close()
        sys.stdout.flush()

    make_tfrecords('train')
    make_tfrecords('test')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        required=True,
        type=str,
        help='Directory containing NTU-RGBD dataset *.skeleton files.')
    parser.add_argument(
        '--output',
        required=False,
        type=str,
        default='.',
        help='Directory to save .tfrecord files.')

    args = parser.parse_args()
    main(dataset_folder=args.dataset, output_folder=args.output)