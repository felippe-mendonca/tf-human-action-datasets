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

  if (x_shape_ndims == 1
      and (expected_shape is None or len(expected_shape) != 1)):
    if tensor_util.is_tensor(x):
      x = array_ops.expand_dims(x, axis=1)
    else:
      x = np.expand_dims(x, 1)
  return x
