from os.path import join, exists
import numpy as np
import pandas as pd
from itertools import combinations

from datasets.montalbanov2.base import MAIN_LINKS, ROOT_JOINT
from datasets.montalbanov2.base import CONNECTED_JOINTS, CONNECTED_BONES


class DataEncoder:
    def __init__(self, dataset_folder, joints_dict):
        avg_distances_file = join(dataset_folder, 'average_distances.csv')
        if not exists(avg_distances_file):
            raise Exception(
                "'average_distances.csv' file doesn't exist on {} folder".format(dataset_folder))

        self._avg_distances = pd.read_csv(avg_distances_file)
        self._joints_dict = joints_dict
        self._pair_joints = list(combinations(self._joints_dict.values(), 2))

        def to_index(tuples):
            return [tuple(map(lambda x: self._joints_dict[x], cj)) for cj in tuples]

        self._connected_joints = to_index(CONNECTED_JOINTS)
        self._connected_bones = to_index(CONNECTED_BONES)

    def normalize_pose(self, pose, joints):
        root_joint = pose[self._joints_dict[ROOT_JOINT], :]
        pose = pose - root_joint
        norm_pose = np.copy(pose)
        for end, start in MAIN_LINKS:
            pos_end, pos_start = self._joints_dict[end], self._joints_dict[start]
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
        return np.array([np.arccos(np.dot(uz, p) / np.linalg.norm(p)) for p in pose])

    def pairwise_distances(self, pose):
        return np.array([np.linalg.norm(pose[i, :] - pose[j, :]) for i, j in self._pair_joints])

    def _vecs_angle(self, v1, v2):
        return np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

    def _body_referential(self, pose):
        ux = pose[self._joints_dict['SHOULDER_RIGHT'], :] - \
             pose[self._joints_dict['SHOULDER_LEFT'], :]
        ux = ux / np.linalg.norm(ux)
        uy = pose[self._joints_dict['HIP_CENTER'], :] - \
             pose[self._joints_dict['SHOULDER_CENTER'], :]
        uy = uy / np.linalg.norm(uy)
        uz = np.cross(ux, uy)
        return ux, uy, uz
    
    @staticmethod
    def normalize(array):
        return (array - array.mean()) / array.std()
