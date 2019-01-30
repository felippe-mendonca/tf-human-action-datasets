from os.path import join, exists
import numpy as np
import pandas as pd
from itertools import combinations
from collections import deque

from datasets.montalbanov2.base import JOINTS, MAIN_JOINTS, ROOT_JOINT
from datasets.montalbanov2.base import MAIN_LINKS
from datasets.montalbanov2.base import CONNECTED_JOINTS, CONNECTED_BONES


class DataEncoder:
    def __init__(self, dataset_folder, smooth_factor=0.8):
        avg_distances_file = join(dataset_folder, 'average_distances.csv')
        if not exists(avg_distances_file):
            raise Exception(
                "'average_distances.csv' file doesn't exist on {} folder".format(dataset_folder))

        self._alpha = smooth_factor
        self._avg_distances = pd.read_csv(avg_distances_file)
        self._main_joints = [JOINTS[x] for x in MAIN_JOINTS]
        self._main_joints_dict = {joint: pos for pos, joint in enumerate(MAIN_JOINTS)}
        all_joints = set(self._main_joints_dict.values())
        root_joint = set([self._main_joints_dict[ROOT_JOINT]])
        # '_nref_joints' is used to exclude referential joint in order do prevent
        # unnecessary computations on bending angles feature, as well as select
        # the positions, velocities and accelerations when creating the features vector.
        self._nref_joints = list(all_joints - root_joint)
        self._pair_joints = list(combinations(self._main_joints_dict.values(), 2))

        def to_index(tuples):
            return [tuple(map(lambda x: self._main_joints_dict[x], cj)) for cj in tuples]

        self._connected_joints = to_index(CONNECTED_JOINTS)
        self._connected_bones = to_index(CONNECTED_BONES)

        self._poses_s = deque(maxlen=2)

    def encode(self, pose):

        pose = self.normalize_pose(pose[self._main_joints, :])
        previous_pose_s = self._poses_s[-1] if len(self._poses_s) > 0 else 0.0
        pose_s = (1 - self._alpha) * previous_pose_s + self._alpha * pose

        if len(self._poses_s) < self._poses_s.maxlen:
            # appends here and at the end just to be
            # consistent with index on forward equations
            self._poses_s.append(pose_s)
            return None

        velocity_s = pose_s - self._poses_s[-1]
        acceleration_s = pose_s - 2 * self._poses_s[-1] + self._poses_s[-2]
        inclination_angles = self.inclination_angles(pose_s)
        azimuth_angles = self.azimuth_angles(pose_s)
        bending_angles = self.bending_angles(pose_s)
        pairwise_distances = self.pairwise_distances(pose_s)

        self._poses_s.append(pose_s)

        vec_features = [
            self.normalize(np.linalg.norm(velocity_s[self._nref_joints, :], axis=1)),
            self.normalize(np.linalg.norm(acceleration_s[self._nref_joints, :], axis=1)),
            np.abs(np.cos(inclination_angles)),
            np.abs(np.sin(inclination_angles)),
            np.abs(np.cos(azimuth_angles)),
            np.abs(np.sin(azimuth_angles)),
            np.abs(np.cos(bending_angles)),
            np.abs(np.sin(bending_angles)),
            self.normalize(pairwise_distances),
        ]

        return np.hstack(vec_features)

    def normalize_pose(self, pose):
        root_joint = pose[self._main_joints_dict[ROOT_JOINT], :]
        pose = pose - root_joint
        norm_pose = np.copy(pose)
        for end, start in MAIN_LINKS:
            pos_end, pos_start = self._main_joints_dict[end], self._main_joints_dict[start]
            d = pose[pos_end] - pose[pos_start]
            r = float(self._avg_distances[end])
            d_ = r * d / np.linalg.norm(d)
            norm_pose[pos_end] = norm_pose[pos_start] + d_

        return norm_pose

    def joint_velocities(self, pose):
        return pose

    def joint_accelerations(self, pose):
        return pose

    def inclination_angles(self, pose):
        angles = []
        for i, j, k in self._connected_joints:
            v1 = pose[i, :] - pose[j, :]
            v2 = pose[k, :] - pose[j, :]
            angles.append(self._vecs_angle(v1, v2))

        return np.array(angles)

    def azimuth_angles(self, pose):
        ux, _, _ = self._body_referential(pose)
        angles = []
        for i, j, k, in self._connected_bones:
            pji = pose[j, :] - pose[i, :]
            pkj = pose[k, :] - pose[j, :]
            a = pji / (np.dot(pkj, pji) / np.dot(pji, pji))
            v1 = ux - a * np.dot(ux, pji)
            v2 = pkj - a * np.dot(pkj, pji)
            angles.append(self._vecs_angle(v1, v2))

        return np.array(angles)

    def bending_angles(self, pose):
        _, _, uz = self._body_referential(pose)
        fpose = pose[self._nref_joints, :]
        return np.array([np.arccos(np.dot(uz, p) / np.linalg.norm(p)) for p in fpose])

    def pairwise_distances(self, pose):
        return np.array([np.linalg.norm(pose[i, :] - pose[j, :]) for i, j in self._pair_joints])

    def _vecs_angle(self, v1, v2):
        return np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

    def _body_referential(self, pose):
        ux = pose[self._main_joints_dict['SHOULDER_RIGHT'], :] - \
             pose[self._main_joints_dict['SHOULDER_LEFT'], :]
        ux = ux / np.linalg.norm(ux)
        uy = pose[self._main_joints_dict['HIP_CENTER'], :] - \
             pose[self._main_joints_dict['SHOULDER_CENTER'], :]
        uy = uy / np.linalg.norm(uy)
        uz = np.cross(ux, uy)
        return ux, uy, uz

    def normalize(self, array):
        return (array - array.mean()) / array.std()
