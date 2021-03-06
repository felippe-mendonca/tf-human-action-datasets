import tensorflow as tf
tf.enable_eager_execution()

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os.path import join
from os import walk
from time import time
from operator import itemgetter
import json

from datasets.tfrecords.features import decode
from models.gesture_localization.encoding import DataEncoder
from models.gesture_localization.model import Model, GestureSpottingState
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


def eval_dataset(model, dataset, options):
    n = 0
    not_gesture_logits, gesture_logits, labels = [], [], []
    gestures_start, gestures_end = [], []
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
        labels.append(np.argmax(label))
        gesture_logits.append(prediction[1])
        not_gesture_logits.append(prediction[0])

    labels = np.array(labels)
    gesture_logits = np.array(gesture_logits)
    not_gesture_logits = np.array(not_gesture_logits)

    # missing end position of last gesture
    if len(gestures_start) > len(gestures_end):
        gesture_width = (n - 1) - gestures_start[-1] + 1
        if gesture_width < options.min_gesture_width:
            del gestures_start[-1]
        else:
            gestures_end.append(n - 1)

    gestures_start = np.array(gestures_start)
    gestures_end = np.array(gestures_end)

    return not_gesture_logits, gesture_logits, labels, gestures_start, gestures_end


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

    def make_dataset(filename):
        dataset = tf.data.TFRecordDataset(filename) \
            .map(decode)                            \
            .map(make_one_hot_label)                \
            .map(standardize_feature)               \
            .batch(1)
        return dataset

    mlp_model_file = model_filename if model_filename.endswith(".hdf5") else None
    random_forest_model_file = model_filename if model_filename.endswith(".sav") else None
    model = Model(mlp_model_file=mlp_model_file,
                  random_forest_model_file=random_forest_model_file,
                  ema_alpha=op.ema_alpha,
                  min_confidence=op.min_confidence,
                  min_gesture_width=op.min_gesture_width,
                  max_undefineds=op.max_n_undefined)

    all_jaccard_indexes = []
    sample_jaccard_indexes = []
    all_took_ms = []
    for file in dataset_files:

        dataset = make_dataset(join(part_folder, file))

        t0 = time()
        result = eval_dataset(model, dataset, op)
        not_gesture_logits, gesture_logits, labels = result[0:3]
        gestures_start, gestures_end = result[3:]
        n = gesture_logits.size
        took_ms = 1000 * ((time() - t0) / n) if n > 0.0 else 0.0
        all_took_ms.append([n, took_ms])

        labels_padded = np.hstack([0, labels, 0])
        dlabels = np.diff(labels_padded)
        labels_start = np.where(dlabels == 1)[0]
        labels_end = np.where(dlabels == -1)[0] - 1

        labels_n = np.zeros(len(labels), dtype=np.uint8)
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
                 sample_jaccard_indexes[-1])

        if display:
            gestures = np.zeros(n)
            for start, end in zip(gestures_start, gestures_end):
                gestures[start:end + 1] = 1

            fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True)

            ax[0].plot(labels, 'k--', linewidth=2.0)
            ax[0].plot(np.array(gestures) + 1, 'g', linewidth=2.0)

            ax[1].fill_between(x=np.arange(gesture_logits.size),
                               y1=gesture_logits,
                               where=gesture_logits > op.min_confidence,
                               color='green',
                               alpha=0.25)

            ax[1].fill_between(x=np.arange(not_gesture_logits.size),
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
    median_pos = int(sorted_indices.size / 2 - 1)
    indices = np.hstack([
        sorted_indices[0:1],
        sorted_indices[median_pos:median_pos + 1],
        sorted_indices[-1:],
    ])

    ranked_jaccard_indexes = sample_jaccard_indexes[indices]
    ranked_samples = itemgetter(*(indices.tolist()))(dataset_files)
    ranked_results = zip(ranked_jaccard_indexes, ranked_samples)
    ranked_results = list(map(lambda x: {'acc': x[0], 'sample': x[1]}, ranked_results))

    all_took_ms = np.array(all_took_ms)
    avg_took_ms = np.dot(all_took_ms[:, 0], all_took_ms[:, 1]) / all_took_ms[:, 0].sum()

    results = {
        "ranked": ranked_results,
        "global": {
            "avg": ji_mean,
            "std": ji_std
        },
        "avg_took_ms": avg_took_ms
    }

    results_filename = options_filename.strip('.json') + '_results.json'
    with open(results_filename, 'w') as f:
        json.dump(results, f, indent=2)

    indices_dict = dict(zip(["worst", "median", "better"], indices))
    for rank, indice in indices_dict.items():
        filename = join(part_folder, dataset_files[indice])
        dataset = make_dataset(filename)
        result = eval_dataset(model, dataset, op)
        not_gesture_logits, gesture_logits, labels = result[0:3]
        gestures_start, gestures_end = result[3:]
        gestures = np.zeros(gesture_logits.size)
        for start, end in zip(gestures_start, gestures_end):
            gestures[start:end + 1] = 1

        data = np.vstack([not_gesture_logits, gesture_logits, labels, gestures]).T
        columns = ['not_gesture_logits', 'gesture_logits', 'labels', 'gestures']
        df = pd.DataFrame(data=data, columns=columns)
        filename = options_filename.strip('.json') + '_{}_ranked_result.csv'.format(rank)
        df.to_csv(path_or_buf=filename, sep=',', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--options',
                        required=True,
                        type=str,
                        help="""Path to options *.json file that matches with a 
        EvalJaccardIndexGestureLocalization protobuf message.""")
    parser.add_argument(
        '--model',
        required=False,
        type=str,
        help=
        """Path to a *.hdf5 file corresponding for a MLP model or a *.sav for a RandomForest model"""
    )
    parser.add_argument("--display", action='store_true')
    args = parser.parse_args()

    main(options_filename=args.options, model_filename=args.model, display=args.display)
