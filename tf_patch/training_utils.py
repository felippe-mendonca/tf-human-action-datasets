# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Training-related utilities.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict
import copy
import math

import numpy as np
import six

import tensorflow as tf
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import losses
from tensorflow.python.keras import metrics as metrics_module
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import weights_broadcast_ops
from tensorflow.python.util import nest


def standardize_single_array(x):
    """Expand data of shape (x,) to (x, 1), unless len(expected_shape)==1."""
    if x is None:
        return None
    if tensor_util.is_tensor(x):
        x_shape_ndims = array_ops.rank(x)
    else:
        x_shape_ndims = len(x.shape)

    if (x_shape_ndims == 1 and (expected_shape is None or len(expected_shape) != 1)):
        if tensor_util.is_tensor(x):
            x = array_ops.expand_dims(x, axis=1)
        else:
            x = np.expand_dims(x, 1)
    return x


"""
And the 'standardize_weights' function.
Github issue: https://github.com/tensorflow/tensorflow/issues/22275
Github PR: https://github.com/tensorflow/tensorflow/pull/23381
Github Commit: https://github.com/tensorflow/tensorflow/commit/c9049de2515cf8643faefa66cc4aea276d390912#diff-4f0d455edc07c640cbb22c8e39a61dd5
"""


def standardize_weights(y, sample_weight=None, class_weight=None, sample_weight_mode=None):
    """Performs sample weight validation and standardization.
  Everything gets normalized to a single sample-wise (or timestep-wise)
  weight array. If both `sample_weight` and `class_weight` are provided,
  the weights are multiplied.
  Arguments:
      y: Numpy array of model targets to be weighted.
      sample_weight: User-provided `sample_weight` argument.
      class_weight: User-provided `class_weight` argument.
      sample_weight_mode: One of `None` or `"temporal"`.
          `"temporal"` indicated that we expect 2D weight data
          that will be applied to the last 2 dimensions of
          the targets (i.e. we are weighting timesteps, not samples).
  Returns:
      A numpy array of target weights, one entry per sample to weight.
  Raises:
      ValueError: In case of invalid user-provided arguments.
  """
    # Iterator may return sample_weight as 1-tuple
    if isinstance(sample_weight, tuple):
        sample_weight = sample_weight[0]
    if sample_weight_mode is not None:
        if sample_weight_mode != 'temporal':
            raise ValueError('"sample_weight_mode '
                             'should be None or "temporal". '
                             'Found: ' + str(sample_weight_mode))
        if len(y.shape) < 3:
            raise ValueError('Found a sample_weight array for '
                             'an input with shape ' + str(y.shape) + '. '
                             'Timestep-wise sample weighting (use of '
                             'sample_weight_mode="temporal") is restricted to '
                             'outputs that are at least 3D, i.e. that have '
                             'a time dimension.')
        if sample_weight is not None and len(sample_weight.shape) != 2:
            raise ValueError('Found a sample_weight array with shape ' + str(sample_weight.shape) +
                             '. '
                             'In order to use timestep-wise sample weighting, '
                             'you should pass a 2D sample_weight array.')
    else:
        if sample_weight is not None and len(sample_weight.shape) != 1:
            raise ValueError('Found a sample_weight array with shape ' + str(sample_weight.shape) +
                             '. '
                             'In order to use timestep-wise sample weights, '
                             'you should specify '
                             'sample_weight_mode="temporal" '
                             'in compile(). If you just mean to use '
                             'sample-wise weights, make sure your '
                             'sample_weight array is 1D.')

    if sample_weight is not None:
        if len(sample_weight.shape) > len(y.shape):
            raise ValueError('Found a sample_weight with shape' + str(sample_weight.shape) + '.'
                             'Expected sample_weight with rank '
                             'less than or equal to ' + str(len(y.shape)))

        if (not tensor_util.is_tensor(sample_weight)
                and y.shape[:sample_weight.ndim] != sample_weight.shape):
            raise ValueError('Found a sample_weight array with shape ' + str(sample_weight.shape) +
                             ' for an input with shape ' + str(y.shape) + '. '
                             'sample_weight cannot be broadcast.')

    # Class weights applied per-sample.
    class_sample_weight = None
    if isinstance(class_weight, dict):
        if len(y.shape) > 2:
            raise ValueError('`class_weight` not supported for ' '3+ dimensional targets.')

        if len(y.shape) == 2:
            if y.shape[1] > 1:
                y_classes = K.argmax(y, axis=1)
                # y_classes = np.argmax(y, axis=1)
            elif y.shape[1] == 1:
                y_classes = np.reshape(y, y.shape[0])
        else:
            y_classes = y

        # class_sample_weight = np.asarray(
        # [class_weight[cls] for cls in y_classes if cls in class_weight])

        keys = list(map(lambda x: tf.cast(x, tf.int32), class_weight.keys()))
        values = list(map(lambda x: tf.cast(x, tf.int32), class_weight.values()))
        key_value = tf.contrib.lookup.KeyValueTensorInitializer(keys, values)
        class_weight_table = tf.contrib.lookup.HashTable(key_value, -1)
        class_sample_weight = class_weight_table.lookup(tf.cast(y_classes, tf.int32))
        class_weight_table.init.run(session=K.get_session())

        # print(K.get_session().run(class_sample_weight))
        # class_sample_weight = np.asarray(
        # [class_weight[cls] for cls in y_classes if cls in class_weight])

        # if len(class_sample_weight) != len(y_classes):
            # subtract the sets to pick all missing classes
            # existing_classes = set(y_classes)
            # existing_class_weight = set(class_weight.keys())
            # raise ValueError('`class_weight` must contain all classes in the data.'
                            #  ' The classes %s exist in the data but not in '
                            #  '`class_weight`.' % (existing_classes - existing_class_weight))

    if class_sample_weight is not None and sample_weight is not None:
        # Multiply weights if both are provided.
        return class_sample_weight * sample_weight
    if sample_weight is not None:
        return sample_weight
    if class_sample_weight is not None:
        return class_sample_weight
    return None
