import os
import re
import numpy as np
import pandas as pd

"""
Different of used on 'ModDrop: Adaptive multi-modal gesture recognition'.
Wrist instead of Hand
"""
MAIN_JOINTS = [
    'HIP_CENTER', 'SHOULDER_CENTER', 'HEAD', 'HIP_RIGHT', 'HIP_LEFT', 'SHOULDER_RIGHT',
    'SHOULDER_LEFT', 'ELBOW_RIGHT', 'ELBOW_LEFT', 'WRIST_RIGHT', 'WRIST_LEFT'
]

"""
The list of links below were created obeying breadth first search (BFS) order,
with HIP_CENTER as root joint. The order of firsts' tuples elements correspond to 
the visited order of BSD algorithm. Each link are created with the predecessor
joint on the tree. Is extremilly importante ensure this order during
normalization process.Follow this order is extremely important during normalization
process. During its process, the start's joint is the second element of the 
tuples (just an implementation detail).
"""
MAIN_LINKS = [
    ('HIP_LEFT', 'HIP_CENTER'),
    ('SHOULDER_CENTER', 'HIP_CENTER'),
    ('HIP_RIGHT', 'HIP_CENTER'),
    ('SHOULDER_RIGHT', 'SHOULDER_CENTER'),
    ('HEAD', 'SHOULDER_CENTER'),
    ('SHOULDER_LEFT', 'SHOULDER_CENTER'),
    ('ELBOW_LEFT', 'SHOULDER_LEFT'),
    ('ELBOW_RIGHT', 'SHOULDER_RIGHT'),
    ('WRIST_LEFT', 'ELBOW_LEFT'),
    ('WRIST_RIGHT', 'ELBOW_RIGHT'),
]

ROOT_JOINT = 'HIP_CENTER' 

JOINTS = {
    'HIP_CENTER': 0,
    'SPINE': 1,
    'SHOULDER_CENTER': 2,
    'HEAD': 3,
    'SHOULDER_LEFT': 4,
    'ELBOW_LEFT': 5,
    'WRIST_LEFT': 6,
    'HAND_LEFT': 7,
    'SHOULDER_RIGHT': 8,
    'ELBOW_RIGHT': 9,
    'WRIST_RIGHT': 10,
    'HAND_RIGHT': 11,
    'HIP_LEFT': 12,
    'KNEE_LEFT': 13,
    'ANKLE_LEFT': 14,
    'FOOT_LEFT': 15,
    'HIP_RIGHT': 16,
    'KNEE_RIGHT': 17,
    'ANKLE_RIGHT': 18,
    'FOOT_RIGHT': 19,
}

LINKS = [
    ('HIP_CENTER', 'SPINE'),
    ('SPINE', 'SHOULDER_CENTER'),
    ('SHOULDER_CENTER', 'HEAD'),
    ('SHOULDER_CENTER', 'SHOULDER_LEFT'),
    ('SHOULDER_LEFT', 'ELBOW_LEFT'),
    ('ELBOW_LEFT', 'WRIST_LEFT'),
    ('WRIST_LEFT', 'HAND_LEFT'),
    ('SHOULDER_CENTER', 'SHOULDER_RIGHT'),
    ('SHOULDER_RIGHT', 'ELBOW_RIGHT'),
    ('ELBOW_RIGHT', 'WRIST_RIGHT'),
    ('WRIST_RIGHT', 'HAND_RIGHT'),
    ('HIP_CENTER', 'HIP_LEFT'),
    ('HIP_LEFT', 'KNEE_LEFT'),
    ('KNEE_LEFT', 'ANKLE_LEFT'),
    ('ANKLE_LEFT', 'FOOT_LEFT'),
    ('HIP_CENTER', 'HIP_RIGHT'),
    ('HIP_RIGHT', 'KNEE_RIGHT'),
    ('KNEE_RIGHT', 'ANKLE_RIGHT'),
    ('ANKLE_RIGHT', 'FOOT_RIGHT'),
]


class Reader:
    def __init__(self, folder):
        self._folder = folder
        _, _, self._sample_files = next(os.walk(self._folder))

        is_sample = re.compile('^(Sample[0-9]{4})_skeleton.csv$')
        valid_sample = lambda x: is_sample.match(x)
        get_sample = lambda x: valid_sample(x).groups()[0]

        self._samples = sorted(map(get_sample, filter(valid_sample, self._sample_files)))
        self._sample_it = -1
        self._pose_csv_columns = np.sort(np.hstack([np.arange(i, 9 * 20, 9) for i in range(3)]))

    def __iter__(self):
        self._sample_it = -1
        return self

    def __next__(self):
        self._sample_it += 1
        if self._sample_it >= len(self._samples):
            raise StopIteration

        sample = self._samples[self._sample_it]
        skeletons_file = os.path.join(self._folder, '{}_skeleton.csv'.format(sample))
        poses = pd.read_csv(skeletons_file, sep=',', usecols=self._pose_csv_columns)
        labels_file = os.path.join(self._folder, '{}_labels.csv'.format(sample))
        labels = pd.read_csv(labels_file, sep=',', names=['class', 'begin', 'end'])
        return sample, poses, labels


class PoseIterator:
    def __init__(self, poses):
        self._poses = poses
        self._row = -1

    def __iter__(self):
        self._row = -1
        return self

    def __next__(self):
        self._row += 1
        if self._row >= self._poses.shape[0]:
            raise StopIteration

        return np.array(self._poses.iloc[self._row, :]).reshape((-1, 3))
