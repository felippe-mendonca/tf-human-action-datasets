import numpy as np
import tensorflow as tf

from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.layers.noise import GaussianDropout, AlphaDropout
from tensorflow.python.keras.layers import Activation
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras.callbacks import Callback

from tensorflow.python.keras.models import load_model as load_keras_model
from sklearn.externals.joblib import load as load_sklearn_model

from models.options.options_pb2 import ActivationFunction
from models.options.options_pb2 import DropoutType
from enum import Enum


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


class GestureSpottingState(Enum):
    RESTING = 1
    START = 2
    PERFORMING = 3
    END = 4
    EARLY_END = 5


class Model:
    def __init__(self,
                 mlp_model_file=None,
                 random_forest_model_file=None,
                 ema_alpha=None,
                 min_confidence=0.5,
                 min_gesture_width=None,
                 max_undefineds=0):

        if mlp_model_file is None and random_forest_model_file is None:
            raise Exception('Must specify at least one model file.')

        if mlp_model_file is not None:
            self._mlp_model = load_keras_model(mlp_model_file, compile=False)
        else:
            self._mlp_model = None

        if random_forest_model_file is not None:
            self._rf_model = load_sklearn_model(random_forest_model_file)
        else:
            self._rf_model = None

        if self._mlp_model is not None and self._rf_model is not None:
            self._predict = self._predict_both_models
        elif self._mlp_model is not None:
            self._predict = self._predict_mlp
        else:
            self._predict = self._predict_rf

        self._apply_ema = (ema_alpha is not None) and (ema_alpha < 1.0)
        self._alpha = ema_alpha
        self._min_confidence = min_confidence
        self._width_validation = min_gesture_width is not None
        self._min_gesture_width = min_gesture_width
        self._max_undefineds = max_undefineds
        self._last_prob_s = 0.0
        self._last_mlp_prob_s = None if ema_alpha is None else 0.0
        self._last_rf_prob_s = None if ema_alpha is None else 0.0
        self._on_gesture = False
        self._undefineds = 0
        self._gesture_width = 0

    def reset(self):
        self._last_prob_s = 0.0
        self._last_mlp_prob_s = 0.0
        self._last_rf_prob_s = 0.0
        self._on_gesture = False
        self._undefineds = 0
        self._gesture_width = 0

    def _update_ema(self, current, last):
        if self._apply_ema:
            return (1 - self._alpha) * last + self._alpha * current
        else:
            return current

    def _predict_mlp(self, x):
        mlp_prob = self._mlp_model.predict_on_batch(x)
        mlp_prob = np.array(mlp_prob)
        mlp_prob_s = self._update_ema(mlp_prob, self._last_mlp_prob_s)
        self._last_mlp_prob_s = mlp_prob_s
        return np.squeeze(mlp_prob_s)

    def _predict_rf(self, x):
        rf_prob = self._rf_model.predict_proba(x)
        rf_prob = np.array(rf_prob)
        rf_prob_s = self._update_ema(rf_prob, self._last_rf_prob_s)
        self._last_rf_prob_s = rf_prob_s
        return np.squeeze(rf_prob_s)

    def _predict_both_models(self, x):
        mlp_prob = self._mlp_model.predict_on_batch(x)
        rf_prob = self._rf_model.predict_proba(x)
        mlp_prob, rf_prob = np.array(mlp_prob), np.array(rf_prob)

        mlp_prob_s = self._update_ema(mlp_prob, self._last_mlp_prob_s)
        rf_prob_s = self._update_ema(rf_prob, self._last_rf_prob_s)
        self._last_mlp_prob_s, self._last_rf_prob_s = mlp_prob_s, rf_prob_s

        if not np.all(mlp_prob_s < self._min_confidence):
            return np.squeeze(mlp_prob_s)
        if not np.all(rf_prob_s < self._min_confidence):
            return np.squeeze(rf_prob_s)
        if np.max(mlp_prob_s) > np.max(rf_prob_s):
            return np.squeeze(mlp_prob_s)
        else:
            return np.squeeze(rf_prob_s)

    def predict(self, x):
        if isinstance(x, tf.Tensor):
            x = np.array(x)
        if not isinstance(x, np.ndarray):
            raise TypeError("'x' can only be either tf.Tensor or np.ndarray")

        self._last_prob_s = self._predict(x)
        return self._last_prob_s

    def spot(self):
        is_not_gesture, is_gesture = tuple(self._last_prob_s > self._min_confidence)
        is_undefined = (not is_gesture) and (not is_not_gesture)

        if not self._on_gesture and is_gesture:
            self._on_gesture = True
            self._undefineds = 0
            self._gesture_width = 1
            return GestureSpottingState.START
        elif self._on_gesture:
            undefineds_limit_reached = False
            if is_undefined:
                self._undefineds += 1
                undefineds_limit_reached = self._undefineds == self._max_undefineds 
            if is_not_gesture or undefineds_limit_reached:
                self._on_gesture = False
                if self._gesture_width < self._min_gesture_width:
                    return GestureSpottingState.EARLY_END
                else:
                    return GestureSpottingState.END
            if is_gesture:
                self._gesture_width += 1
                self._undefineds = 0
                return GestureSpottingState.PERFORMING

        return GestureSpottingState.RESTING