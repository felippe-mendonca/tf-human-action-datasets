import tensorflow as tf
tf.enable_eager_execution()

import argparse
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
from os import walk
from time import time
from operator import itemgetter
import json

from datasets.tfrecords.features import decode
from models.gesture_localization.encoding import DataEncoder
from models.gesture_localization.model import EnsembleModel, GestureSpottingState
from models.options.options_pb2 import EvalJaccardIndexGestureLocalization, Datasets, DatasetPart
from models.options.utils import load_options
from utils.logger import Logger
from sklearn.metrics import jaccard_similarity_score
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

tf.logging.set_verbosity(tf.logging.INFO)
log = Logger('EvalJaccardIndex')


def main(options_filename, model_filename, display):
    op = load_options(options_filename, EvalJaccardIndexGestureLocalization)

    dataset_name = Datasets.Name(op.dataset).lower()
    dataset_part = DatasetPart.Name(op.dataset_part).lower()
    dataset_folder = join(op.storage.datasets_folder, dataset_name)
    dataset_tfrecords_folder = join(dataset_folder, '{}_tfrecords'.format(dataset_name))
    part_folder = join(dataset_tfrecords_folder, dataset_part)
    _, _, dataset_files = next(walk(part_folder))
    dataset_files.sort()

    encoder = DataEncoder(dataset_folder=dataset_folder)
    mean_data, std_data = encoder.get_dataset_stats()
    mean_data, std_data = tf.constant(mean_data), tf.constant(std_data)

    def standardize_feature(feature, label):
        feature = (feature - mean_data) / std_data
        return tf.cast(feature, tf.float32), label

    def make_one_hot_label(feature, label):
        return feature, tf.one_hot(label, 2)

    model = EnsembleModel(
        mlp_model_file=model_filename,
        ema_alpha=op.ema_alpha,
        min_confidence=op.min_confidence,
        min_gesture_width=op.min_gesture_width,
        max_undefineds=op.max_n_undefined)

    all_jaccard_indexes = []
    sample_jaccard_indexes = []
    for file in dataset_files:
        dataset = tf.data.TFRecordDataset(join( part_folder, file)) \
            .map(decode)                                            \
            .map(make_one_hot_label)                                \
            .map(standardize_feature)                               \
            .batch(1)

        n = 0
        not_gesture_logits, gesture_logits, labels_array = [], [], []
        gestures_start, gestures_end = [], []
        t0 = time()
        model.reset()
        for features, label in dataset.make_one_shot_iterator():
            prediction = model.predict(features)
            spotting_state = model.spot()

            if spotting_state == GestureSpottingState.START:
                gestures_start.append(n)
            elif spotting_state == GestureSpottingState.END:
                gestures_end.append(n - 1)
            elif spotting_state == GestureSpottingState.EARLY_END:
                del gestures_start[-1]

            n += 1
            labels_array.append(np.argmax(label))
            if display:
                gesture_logits.append(prediction[1])
                not_gesture_logits.append(prediction[0])

        took_ms = 1000 * ((time() - t0) / n) if n > 0.0 else 0.0

        gesture_logits = np.array(gesture_logits)
        not_gesture_logits = np.array(not_gesture_logits)

        # missing end position of last gesture
        if len(gestures_start) > len(gestures_end):
            gesture_width = (n - 1) - gestures_start[-1] + 1
            if gesture_width < op.min_gesture_width:
                del gestures_start[-1]
            else:
                gestures_end.append(n - 1)

        labels_padded = np.array([0] + labels_array + [0], dtype=np.int8)
        dlabels = np.diff(labels_padded)
        labels_start = np.where(dlabels == 1)[0]
        labels_end = np.where(dlabels == -1)[0] - 1

        labels_n = np.zeros(len(labels_array), dtype=np.uint8)
        for start, end, val in zip(labels_start, labels_end, range(1, labels_start.size + 1)):
            labels_n[start:end + 1] = val

        jaccard_indexes = []
        for start, end in zip(gestures_start, gestures_end):
            labels_intersections = np.unique(labels_n[start:(end + 1)])
            labels_intersections = labels_intersections[labels_intersections > 0]
            if labels_intersections.size == 0:
                jaccard_indexes.append(0.0)
                continue

            for label_n in labels_intersections:
                l_start, l_end = labels_start[label_n - 1], labels_end[label_n - 1]
                min_pos = min(l_start, start)
                max_pos = max(l_end, end)
                mask_width = max_pos - min_pos + 1

                label_mask = np.zeros(mask_width, dtype=np.bool)
                label_mask[(l_start - min_pos):(l_end + 1 - min_pos)] = True
                gesture_mask = np.zeros(mask_width, dtype=np.bool)
                gesture_mask[(start - min_pos):(end + 1 - min_pos)] = True
                jaccard_index = jaccard_similarity_score(label_mask, gesture_mask)
                jaccard_indexes.append(jaccard_index)

            all_jaccard_indexes.extend(jaccard_indexes)

        sample_jaccard_indexes.append(np.array(jaccard_indexes).mean())
        log.info("{} | took={:.2f}ms, jaccard_index={:.4f}", file, took_ms,
                 np.array(jaccard_indexes).mean())

        if display:
            gestures = np.zeros(n)
            for start, end in zip(gestures_start, gestures_end):
                gestures[start:end + 1] = 1

            fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True)

            ax[0].plot(labels_array, 'k--', linewidth=2.0)
            ax[0].plot(np.array(gestures) + 1, 'g', linewidth=2.0)

            ax[1].fill_between(
                x=np.arange(gesture_logits.size),
                y1=gesture_logits,
                where=gesture_logits > op.min_confidence,
                color='green',
                alpha=0.25)

            ax[1].fill_between(
                x=np.arange(not_gesture_logits.size),
                y1=not_gesture_logits,
                where=not_gesture_logits > op.min_confidence,
                color='red',
                alpha=0.25)

            plt.show()

    results_filename = options_filename.strip('.json') + '_jaccard_indexes.npy'
    all_jaccard_indexes = np.array(all_jaccard_indexes)
    ji_mean, ji_std = all_jaccard_indexes.mean(), all_jaccard_indexes.std()
    log.info("Average Jaccard Index: {:.4f}±{:.4f}", ji_mean, ji_std)
    log.info("Saving Jaccard Indexes on '{}'", results_filename)
    np.save(results_filename, all_jaccard_indexes)

    sample_jaccard_indexes = np.array(sample_jaccard_indexes)
    nonzero_indices = np.flatnonzero(sample_jaccard_indexes)
    nan_indices = np.flatnonzero(np.isnan(sample_jaccard_indexes))
    valid_indices = np.setxor1d(nonzero_indices, nan_indices)
    sorted_indices = valid_indices[np.argsort(sample_jaccard_indexes[valid_indices])]
    middle_pos = int(sorted_indices.size / 2 - 1)
    samples_to_save = 1
    indices = np.hstack([
        sorted_indices[0:samples_to_save],
        sorted_indices[middle_pos:middle_pos + samples_to_save],
        sorted_indices[-samples_to_save:],
    ])

    ranked_jaccard_indexes = sample_jaccard_indexes[indices]
    ranked_samples = itemgetter(*(indices.tolist()))(dataset_files)
    ranked_results = zip(ranked_jaccard_indexes, ranked_samples)
    ranked_results = list(map(lambda x: {'acc': x[0], 'sample': x[1]}, ranked_results))

    ranked_results_filename = options_filename.strip('.json') + '_ranked_results.json'
    with open(ranked_results_filename, 'w') as f:
        json.dump(ranked_results, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--options',
        required=True,
        type=str,
        help="""Path to options *.json file that matches with a 
        EvalJaccardIndexGestureLocalization protobuf message.""")
    parser.add_argument(
        '--model',
        required=True,
        type=str,
        help="""Path to a *.h5 file corresponding to a MLP model.""")
    parser.add_argument("--display", action='store_true')
    args = parser.parse_args()

    main(options_filename=args.options, model_filename=args.model, display=args.display)
