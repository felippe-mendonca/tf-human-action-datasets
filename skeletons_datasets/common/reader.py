import tensorflow as tf


class DatasetReader:
    def __init__(self,
                 batch_size=None,
                 drop_reminder=True,
                 num_epochs=None,
                 perform_shuffle=True,
                 *args,
                 **kwargs):
        self._batch_size = batch_size
        self._drop_reminder = drop_reminder
        self._num_epochs = num_epochs
        self._perform_shuffle = perform_shuffle
        self._filenames = kwargs['filenames']

        self._dataset = tf.data.TFRecordDataset(*args, **kwargs)
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

    def filter(self, function):
        self._dataset = self._dataset.filter(function)
        return self

    def map(self, function):
        self._dataset = self._dataset.map(function)
        return self

    def get_iterator(self):
        if self._dataset_iterator is None:
            self._dataset_iterator = self._dataset.make_one_shot_iterator()
        return self._dataset_iterator

    def get_inputs(self):
        if self._perform_shuffle:
            buffer_size = sum(1 for _ in tf.io.tf_record_iterator(self._filenames))
            self._dataset = self._dataset.shuffle(buffer_size=buffer_size)
        self._dataset = self._dataset.repeat(count=self._num_epochs)
        if self._batch_size is not None:
            self._dataset = self._dataset.batch(self._batch_size, self._drop_reminder)

        self._dataset_iterator = self._dataset.make_one_shot_iterator()
        return self._dataset_iterator.get_next()
