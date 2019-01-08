import tensorflow as tf
tf.enable_eager_execution()

import sys
import json
import argparse
import logging
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import cv2
from collections import OrderedDict
from os.path import join, exists

from utils.logger import Logger
from datasets.tfrecords.features import decode
from datasets.ntu_rgbd.base import ACTION_NAMES, ONE_PERSON_ACTION
from models.skeleton_net.encoding import DataEncoder

log = Logger(name='ntu_rgbd_create_tfrecords')


def main(dataset_folder, dataset_part, class_id):
    if str(class_id) not in ACTION_NAMES:
        classes = sorted(ACTION_NAMES.items(), key=lambda kv: int(kv[0]))
        classes = OrderedDict(filter(lambda kv: int(kv[0]) in ONE_PERSON_ACTION, classes))
        log.critical('Invalid class_id. \n{}', json.dumps(classes, indent=2))

    if dataset_part not in ['train', 'test']:
        log.critical("Invalid dataset_part. Use 'train' or 'test'")

    tfrecord_filename = join(dataset_folder, dataset_part, '{}.tfrecords'.format(class_id))
    if not exists(tfrecord_filename):
        log.critical("Can't find *.tfrecord file\n'{}'", tfrecord_filename)

    dataset = tf.data.TFRecordDataset(filenames=tfrecord_filename)
    dataset = dataset.map(decode)

    encoder = DataEncoder(
        output_shape=[112, 112],
        one_hot=True,
        n_classes=len(ONE_PERSON_ACTION),
        label_offset_to_zero=1)

    dataset = encoder.apply_to_dataset(dataset)
    dataset_it = dataset.make_one_shot_iterator()

    fig = plt.figure()
    for features, label in dataset_it:
        h, w, _ = map(int, features['trunk'].shape)
        white_space = tf.constant(1.0, shape=[h, w / 2, 3])
        trunk = tf.concat([white_space, features['trunk'], white_space], axis=1)
        arms = tf.concat([features['leftArm'], features['rightArm']], axis=1)
        legs = tf.concat([features['leftLeg'], features['rightLeg']], axis=1)
        img_features = tf.concat([trunk, arms, legs], axis=0)
        plt.imshow(img_features, interpolation='none')

        fig.canvas.draw()
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        image = data.reshape(fig.canvas.get_width_height()[::-1] + (3, ))

        action_name = ACTION_NAMES[str(class_id)].upper()
        cv2.imshow(action_name, image)
        key = cv2.waitKey(0)
        if key == ord('q'):
            sys.exit(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--class-id',
        required=True,
        type=int,
        help='Dataset class number id. [{},{}]'.format(ONE_PERSON_ACTION[0],
                                                       ONE_PERSON_ACTION[-1]))
    parser.add_argument(
        '--dataset-folder', required=True, type=str, help="Folder containing *.tfrecord files")

    parser.add_argument(
        '--dataset-part',
        required=False,
        type=str,
        default='train',
        help="Dataset part, can be either 'train' or 'test'")

    args = parser.parse_args()
    main(
        dataset_folder=args.dataset_folder, dataset_part=args.dataset_part, class_id=args.class_id)
