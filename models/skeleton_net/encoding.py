import tensorflow as tf
tf.enable_eager_execution()

from skeletons_datasets.common.joints import JOINTS_MAP

# Body part reference is the first joint of the list
BODY_PARTS = {
    'trunk': ['head', 'neck', 'spine', 'middleSpine', 'baseSpine'],
    'leftArm': ['lShoulder', 'lElbow', 'lWrist', 'lHand', 'lTipHand', 'lThumb'],
    'rightArm': ['rShoulder', 'rElbow', 'rWrist', 'rHand', 'rTipHand', 'rThumb'],
    'leftLeg': ['lHip', 'lKnee', 'lAnkle', 'lFoot'],
    'rightLeg': ['rHip', 'rKnee', 'rAnkle', 'rFoot'],
}


def to_index(input):
    for bp, joints in input.copy().items():
        input[bp] = list(map(lambda x: JOINTS_MAP[x], joints))


to_index(BODY_PARTS)


def make_features(label, positions):
    features = {}
    for part_name, part_joints in BODY_PARTS.items():
        n_k = len(part_joints) - 1  # number of part body joints
        # --- Compute within-part vectors
        # creates start_joint as a list of one element to prevent
        # tf.gather function to reduce one dimension, causing problems
        #  on further tensor operations
        start_joint, other_joints = [part_joints[0]], part_joints[1:]
        p0 = tf.gather(positions, axis=1, indices=start_joint)
        p = tf.gather(positions, axis=1, indices=other_joints)
        Vw = p - p0
        Vw_norm = tf.norm(Vw, axis=0, keepdims=False)

        # --- Compute between-part vectors
        all_joints = list(JOINTS_MAP.values())
        other_body_joints = list(set(all_joints) - set(part_joints))
        p = tf.gather(positions, axis=1, indices=other_body_joints)
        Vb = p - p0
        Vb_norm = tf.norm(Vb, axis=0, keepdims=False)

        # --- Compute Cosine Distance (CD) features
        # For any vector 'v' of Vw and any vector 'u' of Vw U Vb,
        # and u != v, compute their cosine distance.
        cossine_array = []
        sine_array = []
        triangular_inequality_array = []
        w_indices = range(n_k)
        for w_indice in w_indices:
            other_indices = list(set(w_indices) - set([w_indice]))
            v = tf.gather(Vw, axis=1, indices=[w_indice])
            others_v = tf.gather(Vw, axis=1, indices=other_indices)
            u = tf.concat([others_v, Vb], axis=1)

            v_norm = tf.gather(Vw_norm, axis=0, indices=[w_indice])
            others_v_norm = tf.gather(Vw_norm, axis=0, indices=other_indices)
            u_norm = tf.concat([others_v_norm, Vb_norm], axis=0)

            cossine = tf.reduce_sum(u * v, axis=0) / (u_norm * v_norm)
            sine = tf.sqrt(1 - tf.square(cossine))
            triangular_inequality = tf.norm(u + v, axis=0) / (u_norm + v_norm)

            cossine_array.append(cossine)
            sine_array.append(sine)
            triangular_inequality_array.append(triangular_inequality)

        features[part_name] = {
            'cossine': tf.concat(cossine, axis=0),
            'sine': tf.concat(sine, axis=0),
            'triangular_inequality': tf.concat(triangular_inequality, axis=0)
        }

    return label, features


def scale_tensor(tensor, output_range, dtype=None):
    tensor_min, tensor_max = tf.reduce_min(tensor), tf.reduce_max(tensor)
    tensor = (tensor - tensor_min) / (tensor_max - tensor_min)
    range_min, range_max = min(output_range), max(output_range)
    tensor = tensor * (range_max - range_min) + range_min
    if dtype != None and dtype != tensor.dtype:
        tensor = tf.cast(tensor, dtype=dtype)
    return tensor


def scale_features_values(label, features):
    for part_name in features.keys():
        features[part_name]['cossine'] = (features[part_name]['cossine'] + 1.0) / 2.0
        features[part_name]['sine'] = (features[part_name]['sine'] + 1.0) / 2.0

    return label, features


def stack_features(label, features):
    for part_name in features.keys():
        features[part_name] = tf.stack([
            features[part_name]['cossine'],              \
            features[part_name]['sine'],                 \
            features[part_name]['triangular_inequality']
        ], axis=2)

    return label, features


def scale_features_size(label, features):
    for part_name in features.keys():
        features[part_name].set_shape([None, None, None])
        features[part_name] = tf.expand_dims(features[part_name], axis=0)
        features[part_name] = tf.image.resize_bilinear(features[part_name], size=[224, 224])
        features[part_name] = tf.squeeze(features[part_name])

    return label, features


PIPELINE = [make_features, scale_features_values, stack_features, scale_features_size]