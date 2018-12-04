import numpy as np
import tensorflow as tf

dataset_part = 'train'
tfrecord_filename = 'ntu_rgbd.{}.tfrecords'.format(dataset_part)
record_iterator = tf.io.tf_record_iterator(tfrecord_filename)

tf.enable_eager_execution()

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

    return label, positions

for serialized_feature in record_iterator:
    label, positions = decode(serialized_feature)
    print(label, positions)
    break

