import tensorflow as tf
tf.enable_eager_execution()

from skeletons_datasets.common.reader import DatasetReader
from skeletons_datasets.common.joints import JOINTS_MAP

dataset_part = 'train'
tfrecord_filename = 'ntu_rgbd.{}.tfrecords'.format(dataset_part)
dataset = DatasetReader(tfrecord_filename)

# Body part reference is the first joint of the list
body_parts = {
    'trunk': ['head', 'neck', 'spine', 'middleSpine', 'baseSpine'],
    'leftArm': ['lShoulder', 'lElbow', 'lWrist', 'lHand', 'lTipHand', 'lThumb'],
    'rightArm': ['rShoulder', 'rElbow', 'rWrist', 'rHand', 'rTipHand', 'rThumb'],
    'leftLeg': ['lHip', 'lKnee', 'lAnkle', 'lFoot'],
    'rightLeg': ['rHip', 'rKnee', 'rAnkle', 'rFoot'],
}

# Vectors to compute Normalized Magnitude features
nm_reference_vecs = {
    'trunk': ['neck', 'spine'],
    'leftArm': ['lShoulder', 'lElbow'],
    'rightArm': ['rShoulder', 'rElbow'],
    'leftLeg': ['lHip', 'lKnee'],
    'rightLeg': ['rHip', 'rKnee'],
}


def to_index(input):
    for bp, joints in input.copy().items():
        input[bp] = list(map(lambda x: JOINTS_MAP[x], joints))


to_index(body_parts)
to_index(nm_reference_vecs)


def make_features(label, positions):

    t = positions.shape[2]  # number of samples over time
    n = len(JOINTS_MAP)  # number of body joints
    features = {'CD': {}, 'NM': {}}
    for part_name, part_joints in body_parts.items():
        n_k = len(part_joints) - 1  # number of part body joints
        # --- Compute within-part vectors
        # creates start_joint as a list of one element to prevent
        # tf.gather function to reduce one dimension, causing problems
        #  on further tensor operations
        start_joint, other_joints = [part_joints[0]], part_joints[1:]
        p0 = tf.gather(positions, axis=1, indices=start_joint)
        p = tf.gather(positions, axis=1, indices=other_joints)
        Vw = p - p0

        # --- Compute between-part vectors
        all_joints = list(JOINTS_MAP.values())
        other_body_joints = list(set(all_joints) - set(part_joints))
        p = tf.gather(positions, axis=1, indices=other_body_joints)
        Vb = p - p0

        # --- Compute Cosine Distance (CD) features
        # For any vector 'v' of Vw and any vector 'u' of Vw U Vb,
        # and u != v, compute their cosine distance.
        cd_array = []
        w_indexes = range(n_k - 1)
        for w_index in w_indexes:
            other_indexes = list(set(w_indexes) - set([w_index]))
            v_w = tf.gather(Vw, axis=1, indices=[w_index])
            others_v_w = tf.gather(Vw, axis=1, indices=other_indexes)
            vectors = tf.concat([others_v_w, Vb], axis=1)

            v_w = tf.math.l2_normalize(v_w, axis=0)
            vectors = tf.math.l2_normalize(vectors, axis=0)
            cd = tf.reduce_sum(v_w * vectors, axis=0)
            cd_array.append(cd)

        CD = tf.concat(cd_array, axis=0)
        features['CD'][part_name] = CD

        # --- Compute Normalized Magnitude (NM) features
        # For any vector 'u' of Vw U Vb, NM is defined as
        # the ratio between |u| and |u_0|, which is
        j0, j1 = nm_reference_vecs[part_name]
        u_0 = tf.gather(positions, axis=1, indices=[j1]) - \
              tf.gather(positions, axis=1, indices=[j0])

        NM = tf.norm(tf.concat([Vw, Vb], axis=1), axis=0) / tf.norm(u_0, axis=0)
        features['NM'][part_name] = NM

    return label, positions, features


dataset.map(make_features)

label, positions, features = dataset.get_inputs()
print(label, positions)
