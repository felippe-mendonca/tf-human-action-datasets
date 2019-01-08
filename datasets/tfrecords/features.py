import tensorflow as tf


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def decode(serialized_feature):
    features = tf.parse_single_example(
        serialized_feature,
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'positions': tf.FixedLenFeature([], tf.string),
            'shape': tf.FixedLenFeature([], tf.string)
        })

    label = tf.cast(features['label'], tf.int64)
    shape = tf.decode_raw(features['shape'], tf.int32)
    positions = tf.decode_raw(features['positions'], tf.float64)
    positions = tf.reshape(positions, shape)

    return positions, label
