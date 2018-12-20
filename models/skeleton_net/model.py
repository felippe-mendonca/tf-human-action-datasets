import tensorflow as tf
from tensorflow.keras.models import Model, Sequential, clone_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Add
from tensorflow.python.keras.layers.core import Lambda
from tensorflow.python.keras.layers import Multiply
from tensorflow.keras import Input

from ..base.vgg16 import VGG16

from tensorflow.python.keras.utils.vis_utils import plot_model

# https://github.com/titu1994/Keras-DualPathNetworks
# https://www.pyimagesearch.com/2018/06/04/keras-multiple-outputs-and-multiple-losses/
# https://towardsdatascience.com/bisenet-for-real-time-segmentation-part-iii-f2b40ba4e177
# https://stackoverflow.com/questions/52509328/is-it-possible-in-keras-to-have-an-input-shape-of-width-and-height-32x32?rq=1
# https://flyyufelix.github.io/2016/10/08/fine-tuning-in-keras-part2.html:w

# personalized layer
# https://github.com/keras-team/keras/issues/7736

# visualize
# https://machinelearningmastery.com/visualize-deep-learning-neural-network-model-keras/

# attention
# https://github.com/philipperemy/keras-attention-mechanism


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
        x = Dense(20, name=prefix_name + 'attention_fc1')(x)
        x = Activation('relu', name=prefix_name + 'attention_relu1')(x)
        x = Dense(1, name=prefix_name + 'attention_fc2')(x)

        return Model(inputs=vgg16_model.input, outputs=x)

    inputs = Input(shape=input_shape + (N_PARALLEL_PATHS,), name='input')
    inputs = Lambda(lambda t: tf.unstack(t, axis=-1), name='main_path/unstack1')(inputs)

    parallel_path = []
    parallel_path_outputs = []
    for i in range(N_PARALLEL_PATHS):
        parallel_path.append(make_parallel_path(inputs[i], prefix_name='path{}/'.format(i)))
        parallel_path_outputs.append(parallel_path[i].output)

    x_attention = Concatenate(axis=1)(parallel_path_outputs)
    x_attention = Activation('softmax', name='main_path/attention_softmax1')(x_attention)

    attention_wights = Lambda(
        lambda t: tf.split(value=t, axis=1, num_or_size_splits=N_PARALLEL_PATHS))(x_attention)

    with_attention = []
    for i in range(N_PARALLEL_PATHS):
        name = 'path{}/flatten1'.format(i)
        x = parallel_path[i].get_layer(name).output
        with_attention.append(
            Multiply(name='path{}/attention_mul1'.format(i))([attention_wights[i], x]))

    x = Add()(with_attention)
    x = Dense(N_CLASSES)(x)
    x = Activation('softmax', name='main_path/softmax1')(x)

    return Model(inputs=parallel_path[0].input, outputs=x)


model = make_model(input_shape=(112, 112, 3), n_parallel_paths=2, n_classes=48)
plot_model(model, to_file='model.png', show_shapes=True)