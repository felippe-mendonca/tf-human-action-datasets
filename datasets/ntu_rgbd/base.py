from os import walk
from os.path import join, realpath, dirname
import re
import json
import numpy as np
from collections import defaultdict


ONE_PERSON_ACTION = range(1, 50)
TRAINING_IDS = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38]

actions_filename = join(dirname(realpath(__file__)),'action_names.json')
with open(actions_filename, 'r') as f:
    ACTION_NAMES = json.load(f)

class Loader:
    def __init__(self, folder, load_positions=True, load_headings=False):
        self.folder = folder
        self.load_positions = load_positions
        self.load_headings = load_headings
        self.filename_pattern = re.compile(
            'S{n:}C{n:}P{n:}R{n:}A{n:}.skeleton'.format(n='([0-9]{3})'))

        missing_skeletons_file = join(dirname(realpath(__file__)), 'missing_skeletons')
        with open(missing_skeletons_file, 'r') as f:
            self.missing_skeletons = list(map(lambda x: x.strip(), f.readlines()))

    def list_files(self, only_valids=True, filter_one_person_classes=True,
                   dataset_part='training'):
        _, _, files = next(walk(self.folder))
        skeletons_files = list(filter(lambda x: x.endswith('.skeleton'), files))
        if only_valids:
            skeletons_files = list(filter(self.is_valid_file, skeletons_files))

        if filter_one_person_classes:

            def _one_person(filename):
                metadata = self.get_metadata(filename)
                if metadata is None:
                    return False
                return metadata['action'] in ONE_PERSON_ACTION

            skeletons_files = list(filter(_one_person, skeletons_files))

        def _dataset_part(filename):
            metadata = self.get_metadata(filename)
            if metadata is None:
                return False
            return metadata['performer'] in TRAINING_IDS

        if dataset_part.lower() == 'train':
            skeletons_files = list(filter(_dataset_part, skeletons_files))
        elif dataset_part.lower() == 'test':
            skeletons_files = list(filter(lambda x: not _dataset_part(x), skeletons_files))

        return skeletons_files

    def load_from_file(self, filename):
        if not self.is_valid_file(filename):
            return None

        metadata = self.get_metadata(filename)
        if metadata is None:
            return None

        filepath = join(self.folder, filename)
        with open(filepath, 'r') as f:
            frame_count = int(f.readline())
            bodies = defaultdict(lambda: defaultdict(list))

            for fc in range(frame_count):
                body_count = int(f.readline())

                for bc in range(body_count):
                    numbers = f.readline().split(' ')
                    body_id = int(numbers[0])
                    joint_count = int(f.readline())
                    joints_positions = np.empty((3, joint_count), dtype=np.float)
                    joints_headings = np.empty((4, joint_count), dtype=np.float)

                    for jc in range(joint_count):
                        joint_info = f.readline().split(' ')
                        x, y, z = map(float, joint_info[:3])
                        joints_positions[:, jc] = np.array([x, y, z])
                        hw, hx, hy, hz = map(float, joint_info[7:11])
                        joints_headings[:, jc] = np.array([hw, hx, hy, hz])

                    if self.load_positions:
                        bodies[body_id]['positions'].append(joints_positions)
                    if self.load_headings:
                        bodies[body_id]['headings'].append(joints_headings)

            return metadata, bodies

    def get_metadata(self, filename):
        matches = self.filename_pattern.search(filename)
        if matches is None:
            return None

        setup, camera, performer, replication, action = map(int, matches.groups())
        metadata = {
            'setup': setup,
            'camera': camera,
            'performer': performer,
            'replication': replication,
            'action': action
        }
        return metadata

    def is_valid_file(self, filename):
        return filename.split('.')[0] not in self.missing_skeletons