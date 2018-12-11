import sys
import logging
import argparse
from os.path import join
import numpy as np
import tensorflow as tf

from skeletons_datasets.tfrecords.features import bytes_feature, int64_feature
from skeletons_datasets.ntu_rgbd.base import Loader

FORMAT = '[%(asctime)-15s] %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=FORMAT)
log = logging.getLogger('ntu_rgbd_create_tfrecords')


def main(dataset_folder, output_folder):
    loader = Loader(folder=dataset_folder, load_headings=False)

    def make_tfrecords(dataset_part):
        tfrecord_filename = join(output_folder, 'ntu_rgbd.{}.tfrecords'.format(dataset_part))
        writer = tf.python_io.TFRecordWriter(tfrecord_filename)

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
            writer.write(example.SerializeToString())

            if n % 100 == 0:
                log.info('[{}] {}/{}'.format(tfrecord_filename, n, len(files)))

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