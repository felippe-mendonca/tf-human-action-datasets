import tensorflow as tf
tf.enable_eager_execution()

from skeletons_datasets.common.reader import DatasetReader
from models.skeleton_net.encoding import DataEncoder

g = tf.Graph()
with g.as_default():

    dataset_part = 'train'
    tfrecord_filename = 'ntu_rgbd.{}.tfrecords'.format(dataset_part)
    dataset = DatasetReader(tfrecord_filename, batch_size=10)
    encoder = DataEncoder(output_shape=[112, 112])
    encoder.apply_to_dataset(dataset)

    label, features = dataset.get_inputs()

    with tf.Session() as sess:
        writer = tf.summary.FileWriter('logs', sess.graph)

        label, features = sess.run([label, features])
        for part_name, tensor in features.items():
            print(part_name, tensor.shape)
        
        writer.close()
