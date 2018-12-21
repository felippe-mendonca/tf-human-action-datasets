import tensorflow as tf

from tensorflow.python.keras import backend as K
from tensorflow.python.keras import Input
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dot
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.layers import Activation
from tensorflow.python.keras.layers import Concatenate
from tensorflow.python.keras.layers import AveragePooling2D
from tensorflow.python.keras.layers.core import Lambda

from ..base.vgg16 import VGG16

from tensorflow.python.keras.utils.vis_utils import plot_model


def make_model(input_shape, n_parallel_paths, n_classes):

    N_PARALLEL_PATHS = n_parallel_paths
    N_CLASSES = n_classes

    def make_parallel_path(input_tensor, prefix_name=''):
        # use None instead of 'imagenet' to random initialization
        vgg16_model = VGG16(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape,
            input_tensor=input_tensor,
            prefix_name=prefix_name)

        block3_pool = vgg16_model.get_layer(prefix_name + 'block3_pool').output

        pool_size = block3_pool.shape[1:3]
        x = AveragePooling2D(pool_size=pool_size, name=prefix_name + 'avg_pool')(block3_pool)
        x = Flatten(name=prefix_name + 'flatten1')(x)
        x = Dense(
            20,
            use_bias=False,
            kernel_initializer='random_uniform',
            name=prefix_name + 'attention_fc1')(x)
        x = Activation('relu', name=prefix_name + 'attention_relu1')(x)
        x = Dense(
            1,
            use_bias=False,
            kernel_initializer='random_uniform',
            name=prefix_name + 'attention_fc2')(x)

        return Model(inputs=vgg16_model.input, outputs=x)

    inputs = Input(shape=input_shape + (N_PARALLEL_PATHS, ), name='input')
    inputs = Lambda(lambda t: tf.unstack(t, axis=-1), name='input/unstack1')(inputs)

    parallel_path = []
    parallel_path_outputs = []
    for i in range(N_PARALLEL_PATHS):
        parallel_path.append(make_parallel_path(inputs[i], prefix_name='path{}/'.format(i)))
        parallel_path_outputs.append(parallel_path[i].output)

    x_attention = Concatenate(axis=1, name='attention/concat1')(parallel_path_outputs)
    x_attention = Activation('softmax', name='attention/softmax1')(x_attention)
    x_attention = Lambda(
        lambda t: K.expand_dims(t, axis=1), name='attention/expdims1')(x_attention)

    features_list = []
    for i in range(N_PARALLEL_PATHS):
        name = 'path{}/flatten1'.format(i)
        features_list.append(parallel_path[i].get_layer(name).output)

    features = Lambda(lambda t: K.stack(t, axis=2), name='attention/stack1')(features_list)

    x = Dot(axes=2, name='attention/weighted_sum')([x_attention, features])
    x = Flatten(name='output/flatten1')(x)
    x = Dense(N_CLASSES, use_bias=True, kernel_initializer='random_uniform', name='output/fc1')(x)
    x = Activation('softmax', name='output/softmax1')(x)

    return Model(inputs=parallel_path[0].input, outputs=x)
