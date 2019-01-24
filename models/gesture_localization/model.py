from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.layers import Activation
from tensorflow.python.keras.callbacks import Callback


def make_model(n_features, hidden_neurons):
    inputs = Input(shape=(n_features, ), name='features')
    x = Dense(hidden_neurons, use_bias=True, kernel_initializer='glorot_uniform')(inputs)
    x = Activation(activation='relu')(x)
    x = Dropout(rate=0.25)(x)
    x = Dense(2, use_bias=True, kernel_initializer='glorot_uniform')(x)
    outputs = Activation(activation='softmax')(x)
    return Model(inputs, outputs)

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
            