import tensorflow as tf
import numpy as np

from tensorflow.python.keras import backend as K
from tensorflow.python.keras import Input
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dot
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.layers import Activation
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.layers import Concatenate
from tensorflow.python.keras.layers import AveragePooling2D
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras.layers.core import Lambda
from tensorflow.python.keras.callbacks import Callback

from ..base.vgg16 import vgg16
from ..base.exporters import tf_summary_image


def make_model(input_shape, n_parallel_paths, n_classes):

    N_PARALLEL_PATHS = n_parallel_paths
    N_CLASSES = n_classes

    def make_parallel_path(prefix_name='', input_tensor=None):
        # use None instead of 'imagenet' to random initialization
        vgg16_sliced = vgg16(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape,
            input_tensor=input_tensor,
            prefix_name=prefix_name,
            single_block=True,
            slice_at='block3_pool')

        pool_size = (int(vgg16_sliced.output.shape[1]) / 2, int(vgg16_sliced.output.shape[2]) / 2)
        pool_size = tuple(map(int, pool_size))
        x = AveragePooling2D(
            pool_size=pool_size, name=prefix_name + 'avg_pool')(vgg16_sliced.output)
        x = Flatten(name=prefix_name + 'flatten1')(x)
        x = Dense(
            256, use_bias=True, kernel_initializer='random_uniform', name=prefix_name + 'fc1')(x)
        x = BatchNormalization(name=prefix_name + 'bn1')(x)
        x = Activation('relu', name=prefix_name + 'relu1')(x)
        x = Dropout(0.25, name=prefix_name + 'dropout1')(x)

        return Model(inputs=vgg16_sliced.input, outputs=x)

    parallel_path = []
    parallel_path_outputs = []
    for i in range(N_PARALLEL_PATHS):
        parallel_path.append(make_parallel_path(prefix_name='path{}/'.format(i)))
        parallel_path_outputs.append(parallel_path[i].output)

    x = Concatenate(axis=1, name='attention/concat1')(parallel_path_outputs)
    x = Dense(N_CLASSES, use_bias=True, kernel_initializer='random_uniform', name='output/fc1')(x)
    x = Activation('softmax', name='output/softmax1')(x)

    inputs = [p.input for p in parallel_path]
    return Model(inputs=inputs, outputs=x)


class InputsExporter(Callback):
    def __init__(self, features, labels, log_dir='.', period=1, class_labels=None):
        self.features = features
        self.labels = labels
        self.period = period
        self.log_dir = log_dir
        self.class_labels = class_labels
        self.first_batch = None

    def on_batch_end(self, batch, logs=None):
        self.first_batch = self.first_batch or batch
        if (batch - self.first_batch) % self.period == 0:
            logs = logs or {}
            print('[InputsExporter] batch {}'.format(batch))

            labels = K.get_session().run(self.labels)
            labels = np.argmax(labels, axis=1) + 1
            for vgg_path, image_tensor in self.features.items():
                images = K.get_session().run(image_tensor)
                for pos, label in enumerate(labels):
                    image = images[pos, ...] * 255
                    image = tf_summary_image(np.copy(image))
                    tag = vgg_path + '/{}'.format(label)
                    display_name = '' if self.class_labels is None else self.class_labels[str(label)]
                    summary = tf.Summary(value=[
                        tf.Summary.Value(
                            tag=tag,
                            metadata=tf.SummaryMetadata(display_name=display_name),
                            image=image)
                    ])
                    writer = tf.summary.FileWriter(self.log_dir)
                    writer.add_summary(summary, batch)
                    writer.close()
        return
