import re
import sys
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
from datasets.montalbanov2.base import JOINTS, MAIN_JOINTS, ROOT_JOINT
from models.gesture_localization.encoding import DataEncoder

log = Logger(name='montalbanov2_create_tfrecords')


def main(dataset_folder, output_folder):

    main_joints = [JOINTS[x] for x in MAIN_JOINTS]
    main_joints_dict = {joint: pos for pos, joint in enumerate(MAIN_JOINTS)}

    all_joints = set(main_joints_dict.values())
    root_joint = set([main_joints_dict[ROOT_JOINT]])
    features_joints = list(all_joints - root_joint)

    encoder = DataEncoder(dataset_folder, main_joints_dict)

    def make_tfrecords(dataset_part):
        part_folder = join(output_folder, 'montalbanov2_tfrecords', dataset_part)
        if exists(part_folder):
            rmtree(part_folder)
        makedirs(part_folder)

        reader = Reader(folder=join(dataset_folder, dataset_part))
        for sample, poses, labels in reader:

            log.info('[Start] {}', sample)
            tfrecord_filename = join(part_folder, sample + '.tfrecords')
            writer = tf.python_io.TFRecordWriter(tfrecord_filename)

            one_host_labels = np.zeros(poses.shape[0])
            for _, label in labels.iterrows():
                one_host_labels[label['begin']:label['end'] + 1] = 1

            for pose, label in zip(PoseIterator(poses), one_host_labels):
                if np.all(pose == 0.0):
                    continue

                pose = encoder.normalize_pose(pose[main_joints, :], main_joints_dict)

                features_pose = pose[features_joints, :]
                velocities = encoder.joint_velocities(features_pose)
                accelerations = encoder.joint_accelerations(features_pose)
                inclination_angles = encoder.inclination_angles(pose)
                azimuth_angles = encoder.azimuth_angles(pose)
                bending_angles = encoder.bending_angles(features_pose)
                pairwise_distances = encoder.pairwise_distances(pose)

                vec_features = [
                    features_pose.ravel(), velocities.ravel(), accelerations.ravel(), \
                    inclination_angles, azimuth_angles, bending_angles, pairwise_distances
                ]

                vec_features = np.hstack(map(DataEncoder.normalize, vec_features))
                shape = np.array(vec_features.shape, dtype=np.int32)

                feature = {
                    'label': int64_feature(int(label)),
                    'positions': bytes_feature(vec_features.tobytes()),
                    'shape': bytes_feature(shape.tobytes())
                }

                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())

            writer.close()
            sys.stdout.flush()
            log.info('[Done] {}', sample)

    make_tfrecords('train')
    make_tfrecords('validation')


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

    args = parser.parse_args()
    main(dataset_folder=args.dataset, output_folder=args.output)