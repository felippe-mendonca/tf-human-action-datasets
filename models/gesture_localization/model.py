from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.layers import Activation


def make_model(n_features, hidden_neurons):
    inputs = Input(shape=(n_features, ), name='features')
    x = Dense(hidden_neurons, use_bias=True, kernel_initializer='normal')(inputs)
    x = Activation(activation='relu')(x)
    x = Dense(1, use_bias=True, kernel_initializer='normal')(x)
    outputs = Activation(activation='sigmoid')(x)
    return Model(inputs, outputs)
