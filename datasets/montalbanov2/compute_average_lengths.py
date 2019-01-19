import argparse
import numpy as np
import pandas as pd
from os.path import join

from datasets.montalbanov2.base import Reader
from datasets.montalbanov2.base import JOINTS, MAIN_LINKS, ROOT_JOINT


def main(dataset_folder, output_folder):
    train_folder = join(dataset_folder, 'train')
    reader = Reader(folder=train_folder)

    main_joints_id_bsd = [JOINTS[x[0]] for x in MAIN_LINKS]
    root_joint_id = JOINTS[ROOT_JOINT]

    batch_avg_distances = []
    batch_weights = []
    for sample, poses, labels in reader:
        shape = (poses.shape[0], -1, 3)
        poses_array = np.expand_dims(np.array(poses), axis=-1).reshape(shape)
        main_joints = poses_array[:, main_joints_id_bsd, :]
        root_joints = poses_array[:, [root_joint_id], :]
        main_joints = main_joints - root_joints
        main_joints_distances = np.linalg.norm(main_joints, axis=2)
        batch_avg_distances.append(np.mean(main_joints_distances, axis=0))
        batch_weights.append(main_joints_distances.shape[0])

    md = np.array(batch_avg_distances)
    w = np.expand_dims(np.array(batch_weights), axis=-1)
    avg_distances = np.sum(md * w, axis=0) / np.sum(w, axis=0)
    avg_distances = avg_distances / np.linalg.norm(avg_distances)

    avg_distances_df = pd.DataFrame(
        data=np.expand_dims(avg_distances, axis=0), columns=[x[0] for x in MAIN_LINKS])
    filename = join(dataset_folder, 'average_distances.csv')
    avg_distances_df.to_csv(path_or_buf=filename, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""This script computes the average distance
of each main link throughout all train dataset. Afterwards, stores the normalized values on
a file named average_distances.csv on the given dataset directory. The columns of the csv
file are labeled with the first joint name, belonged to each tuple of MAIN_LINKS list.
This list can be found on datasets.montalbanov2.base module.""")
    parser.add_argument(
        '--dataset',
        required=True,
        type=str,
        help='Directory containing MontalbanoV2 dataset with *.csv files.')
    parser.add_argument(
        '--output',
        required=False,
        type=str,
        default='.',
        help='Directory to save average_lengths.csv file.')

    args = parser.parse_args()
    main(dataset_folder=args.dataset, output_folder=args.output)