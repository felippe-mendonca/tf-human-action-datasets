import tensorflow as tf


class DatasetReader:
    def __init__(self, filename, batch_size=None, drop_reminder=True, num_epochs=None):
        self._batch_size = batch_size
        self._drop_reminder = drop_reminder
        self._num_epochs = num_epochs

        self._dataset = tf.data.TFRecordDataset(filename)
        self._dataset = self._dataset.map(self._decode)
        self._dataset_iterator = None

    def _decode(self, serialized_feature):
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

    def map(self, function):
        self._dataset = self._dataset.map(function)
        return self

    def get_iterator(self):
        if self._dataset_iterator is None:
            self._dataset_iterator = self._dataset.make_one_shot_iterator()
        return self._dataset_iterator

    def get_inputs(self):
        self._dataset = self._dataset.repeat(count=self._num_epochs)
        if self._batch_size is not None:
            self._dataset = self._dataset.batch(self._batch_size, self._drop_reminder)

        self._dataset_iterator = self._dataset.make_one_shot_iterator()
        return self._dataset_iterator.get_next()
