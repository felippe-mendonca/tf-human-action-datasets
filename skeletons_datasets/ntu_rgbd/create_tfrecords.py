import sys
import logging
import numpy as np
import tensorflow as tf

from skeletons_datasets.tfrecords.features import bytes_feature, int64_feature
from skeletons_datasets.ntu_rgbd.base import Loader

FORMAT = '[%(asctime)-15s] %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=FORMAT)
log = logging.getLogger('ntu_rgbd_create_tfrecords')

dataset_folder = '/home/felippe/datasets/NTURGB-D/nturgb+d_skeletons/'
loader = Loader(folder=dataset_folder, load_headings=False)

def make_tfrecords(dataset_part):
    tfrecord_filename = 'ntu_rgbd.{}.tfrecords'.format(dataset_part)
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
