from os.path import join
import argparse
import numpy as np
import tensorflow as tf
tf.enable_eager_execution()

from datasets.tfrecords.features import decode
from models.gesture_localization.encoding import DataEncoder
from utils.logger import Logger

log = Logger(name='GestureLocalization T-SNE')


def main(dataset_folder, output_folder):
    encoder = DataEncoder(dataset_folder=dataset_folder)
    mean_data, std_data = encoder.get_dataset_stats()
    mean_data, std_data = tf.constant(mean_data), tf.constant(std_data)

    def standardize_feature(feature, label):
        feature = (feature - mean_data) / std_data
        return feature, label

    parts = ['train', 'validation', 'test']
    for part in parts:
        log.info("Starting '{}' part.", part)

        files = tf.data.Dataset().list_files(
            join(dataset_folder, 'montalbanov2_tfrecords', part, 'Sample*.tfrecords'))
        dataset = tf.data.TFRecordDataset(filenames=files)
        dataset = dataset.map(decode)
        dataset = dataset.map(standardize_feature)
        dataset = dataset.batch(batch_size=100)

        features_filename = join(output_folder, '{}_features.tsv'.format(part))
        labels_filename = join(output_folder, '{}_labels.tsv'.format(part))
        open(features_filename, 'w').close()  # only to ensure that files are initially empty
        open(labels_filename, 'w').close()
        features_file = open(features_filename, 'a')
        labels_file = open(labels_filename, 'a')

        for (features, labels), i in zip(dataset.make_one_shot_iterator(), range(100)):
            features, labels = np.array(features), np.array(labels, dtype=np.uint8)[:, np.newaxis]
            np.savetxt(features_file, features, delimiter='\t', fmt='%.10e')
            np.savetxt(labels_file, labels, delimiter='\t', fmt='%d')
            log.info('[{}][{}/100]', part, i + 1)

        features_file.close()
        labels_file.close()

        log.info("Done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        required=True,
        type=str,
        help=
        "Directory containing MontalbanoV2 tfrecords files, of train, validation and test datasets parts"
    )
    parser.add_argument(
        '--output',
        required=False,
        type=str,
        default='.',
        help="Directory to save tensorboard summaries.")

    args = parser.parse_args()
    main(dataset_folder=args.dataset, output_folder=args.output)