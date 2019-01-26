import re
from os.path import join
import numpy as np
import telegram
import tensorflow as tf
from tensorflow.python.keras.callbacks import Callback
from tensorflow.python.keras import backend as K


class TensorBoardMetrics(Callback):
    def __init__(self, log_dir='./logs', write_graph=True):
        super(TensorBoardMetrics, self).__init__()
        self.log_dir = log_dir
        self.sess = K.get_session()

        self.writer = tf.summary.FileWriter(self.log_dir, graph=self.sess.graph)
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
            elif maybe_validation is not None:
                summary_value.tag = maybe_validation.groups()[0]
                self.validation_writer.add_summary(summary, epoch)
                self.validation_writer.flush()
            else:
                summary_value.tag = name
                self.writer.add_summary(summary, epoch)
                self.writer.flush()

    def on_train_end(self, _):
        self.writer.close()
        self.train_writer.close()
        self.validation_writer.close()


class LearningRateScheduler(Callback):
    """Learning rate scheduler.
    # Arguments
        schedule: a function that takes an epoch index as input
            (integer, indexed from 0) and current learning rate
            and returns a new learning rate as output (float).
        verbose: int. 0: quiet, 1: update messages.
    """

    def __init__(self, schedule, verbose=0):
        super(LearningRateScheduler, self).__init__()
        self.schedule = schedule
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        lr = float(K.get_value(self.model.optimizer.lr))
        try:  # new API
            lr = self.schedule(epoch, lr)
        except TypeError:  # old API for backward compatibility
            lr = self.schedule(epoch)
        if not isinstance(lr, (float, np.float32, np.float64)):
            raise ValueError('The output of the "schedule" function should be float.')
        K.set_value(self.model.optimizer.lr, lr)
        if self.verbose > 0:
            print('\nEpoch %05d: LearningRateScheduler setting learning '
                  'rate to %s.' % (epoch + 1, lr))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)


class TelegramExporter(Callback):
    """
    Callback to monitor your training receiving messages on Telegram.
    To ger your telegram_id, send '/get_my_id' to  @FalconGate_Bot
    """

    def __init__(self, telegram_id, token):
        super(TelegramExporter, self).__init__()

        if not isinstance(telegram_id, int):
            raise Exception("'telegram_id' must be an integer.")

        if not isinstance(token, str):
            raise Exception("Telegram's 'token' must be a string.")

        self.telegram_id = telegram_id
        self.bot = telegram.Bot(token)

    def send_message(self, text):
        try:
            self.bot.send_message(chat_id=self.telegram_id, text=text)
        except Exception as e:
            print("Can't sent message.\n{}.".format(e))

    def on_train_begin(self, logs={}):
        text = 'Start training model {}.'.format(self.model.name)
        self.send_message(text)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        text = 'Model {}\nEpoch {}.\n'.format(self.model.name, epoch)
        for k, v in sorted(logs.items()):
            text += '{}: {:.4f}\n'.format(k, v)
        self.send_message(text)
