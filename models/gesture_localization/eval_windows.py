import tensorflow as tf
tf.enable_eager_execution()

import numpy as np
from tensorflow.python.keras.models import load_model
import matplotlib.pyplot as plt

from os.path import join
from os import walk
from time import time

from datasets.tfrecords.features import decode
from models.gesture_localization.encoding import DataEncoder
"""
Monkey patch to fix issue on 'standardize_single_array' function.
GitHub issue: https://github.com/tensorflow/tensorflow/issues/24520
GitHub PR: https://github.com/tensorflow/tensorflow/pull/24522

IMPORTANT: this patch must be placed after all tensorflow imports to 
           prevent function redefinitions, invalidating this patch.
"""
from tensorflow.python.keras import engine
from tf_patch.training_utils import standardize_single_array as _standardize_single_array
engine.training_utils.standardize_single_array = _standardize_single_array

model_path = 'model-0131-0.9222.hdf5'
model = load_model(model_path, compile=False)

dataset_folder = '/home/felippe/datasets/montalbanov2/'
part_folder = join(dataset_folder, 'montalbanov2_tfrecords/test/')
_, _, dataset_files = next(walk(part_folder))

encoder = DataEncoder(dataset_folder=dataset_folder)
mean_data, std_data = encoder.get_dataset_stats()
mean_data, std_data = tf.constant(mean_data), tf.constant(std_data)


def standardize_feature(feature, label):
    feature = (feature - mean_data) / std_data
    return feature, label


def make_one_hot_label(feature, label):
    return feature, tf.one_hot(label, 2)


for file in sorted(dataset_files):
    dataset = tf.data.TFRecordDataset(join(
        part_folder, file)).map(decode).map(make_one_hot_label).map(standardize_feature).batch(1)
    n = 0
    t0 = time()
    not_gesture_logits = []
    gesture_logits = []
    labels_array = []
    alpha = 0.5
    for features, label in dataset.make_one_shot_iterator():
        prediction = np.array(model.predict_on_batch(tf.cast(features, tf.float32)))
        p, l = np.argmax(prediction), np.argmax(label)

        last_ngl = not_gesture_logits[-1] if len(not_gesture_logits) > 0 else 0.0
        last_gl = gesture_logits[-1] if len(gesture_logits) > 0 else 0.0
        ngl = (1 - alpha) * last_ngl + alpha * prediction[0][0]
        gl = (1 - alpha) * last_gl + alpha * prediction[0][1]

        gesture_logits.append(gl)
        not_gesture_logits.append(ngl)
        labels_array.append(np.argmax(label))

        n += 1

    t1 = time()
    print(file, 1000 * ((t1 - t0) / n))

    min_confidence = 0.7
    gesture_logits = np.array(gesture_logits)
    not_gesture_logits = np.array(not_gesture_logits)

    fig, ax = plt.subplots()
    ax.plot(labels_array, 'k--', linewidth=3.0)

    ax.fill_between(
        x=np.arange(gesture_logits.size),
        y1=gesture_logits,
        where=gesture_logits > min_confidence,
        color='green',
        alpha=0.5)

    ax.fill_between(
        x=np.arange(not_gesture_logits.size),
        y1=not_gesture_logits,
        where=not_gesture_logits > min_confidence,
        color='red',
        alpha=0.5)

    plt.show()