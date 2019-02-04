from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.layers.noise import GaussianDropout, AlphaDropout
from tensorflow.python.keras.layers import Activation
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras.callbacks import Callback

from models.options.options_pb2 import ActivationFunction
from models.options.options_pb2 import DropoutType


def make_model(n_features, hidden_layers=None, print_summary=False, name=None):
    inputs = Input(shape=(n_features, ), name='features')
    x = inputs
    if hidden_layers is not None:
        for layer in hidden_layers:
            x = Dense(units=layer.units, use_bias=True, kernel_initializer='glorot_uniform')(x)

            if layer.HasField('dropout'):
                if layer.dropout.type == DropoutType.Value('STANDARD'):
                    x = Dropout(rate=layer.dropout.rate)(x)
                elif layer.dropout.type == DropoutType.Value('GAUSSIAN'):
                    x = GaussianDropout(rate=layer.dropout.rate)(x)
                elif layer.dropout.type == DropoutType.Value('ALPHA'):
                    x = AlphaDropout(rate=layer.dropout.rate)(x)
                else:
                    raise TypeError('Invalid DropoutType specified.')

            activation = ActivationFunction.Name(layer.activation).lower()
            x = Activation(activation=activation)(x)

            if layer.batch_normalization:
                x = BatchNormalization()(x)

    x = Dense(2, use_bias=True, kernel_initializer='glorot_uniform')(x)
    outputs = Activation(activation='softmax')(x)
    model = Model(inputs, outputs, name=name)

    if print_summary:
        model.summary()

    return model


class EvalTrainDataset(Callback):
    def __init__(self, model, train_dataset, steps, interval=1):
        self.model = model
        self.train_dataset = train_dataset
        self.steps = steps
        self.interval = interval

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            hist = self.model.evaluate(x=self.train_dataset, steps=self.steps)
            logs['test_loss'] = hist[0]
            logs['test_acc'] = hist[1]
