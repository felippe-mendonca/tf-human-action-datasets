import tensorflow as tf


class DatasetReader:
    def __init__(self,
                 batch_size=None,
                 drop_reminder=True,
                 shuffle_size=None,
                 prefetch_size=None,
                 *args,
                 **kwargs):
        self._batch_size = batch_size
        self._drop_reminder = drop_reminder
        self._shuffle_size = shuffle_size
        self._prefetch_size = prefetch_size
        self._final_setup = False

        self._dataset = tf.data.TFRecordDataset(*args, **kwargs)
        self._dataset = self._dataset.map(self._decode)

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
        return self._dataset.make_one_shot_iterator()

    def get_inputs(self):
        if not self._final_setup:
            if self._shuffle_size is not None:
                self._dataset = self._dataset.shuffle(buffer_size=self._shuffle_size)

            if self._batch_size is not None:
                self._dataset = self._dataset.batch(self._batch_size, self._drop_reminder)

            if self._prefetch_size is not None:
                self._dataset = self._dataset.prefetch(self._prefetch_size)
            
            self._final_setup = True

        return self._dataset.make_one_shot_iterator().get_next()
