import re
from os.path import join
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.callbacks import Callback
from tensorflow.python.keras import backend as K


class TensorBoardMetrics(Callback):
    def __init__(self, log_dir='./logs', write_graph=True):
        self.log_dir = log_dir
        self.sess = K.get_session()

        self.graph_writer = tf.summary.FileWriter(self.log_dir, graph=self.sess.graph)
        self.train_writer = tf.summary.FileWriter(join(self.log_dir, 'train'))
        self.validation_writer = tf.summary.FileWriter(join(self.log_dir, 'validation'))

        self.train_re = re.compile('^(acc|loss)$')
        self.validation_re = re.compile('^val_(acc|loss)$')

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item() if isinstance(value, np.ndarray) else value

            maybe_train = self.train_re.match(name)
            maybe_validation = self.validation_re.match(name)
            if maybe_train is not None:
                summary_value.tag = maybe_train.groups()[0]
                self.train_writer.add_summary(summary, epoch)
                self.train_writer.flush()
            if maybe_validation is not None:
                summary_value.tag = maybe_validation.groups()[0]
                self.validation_writer.add_summary(summary, epoch)
                self.validation_writer.flush()

    def on_train_end(self, _):
        self.train_writer.close()
        self.validation_writer.close()
