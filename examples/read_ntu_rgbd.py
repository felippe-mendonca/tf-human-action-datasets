import tensorflow as tf
tf.enable_eager_execution()

from skeletons_datasets.common.reader import DatasetReader

dataset_part = 'train'
tfrecord_filename = 'ntu_rgbd.{}.tfrecords'.format(dataset_part)
reader = DatasetReader(tfrecord_filename)

label, positions = reader.get_inputs()
print(label, positions)

iterator = reader.get_iterator()
label, positions = iterator.get_next()
print(label, positions)
