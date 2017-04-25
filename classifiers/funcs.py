import numpy as np
from sklearn.preprocessing import scale


def get_labels_numerical(labels_txt):
    labels_num = [get_label_numerical(label) for label in labels_txt]
    return np.array(labels_num)


def get_label_numerical(label):
    if label == 'positive':
        label_num = 1
    elif label == 'negative':
        label_num = -1
    else:
        label_num = 0
    return label_num


def scale_vector(vec):
    return scale(vec, copy=True)


def regularize_hor(features):
    for i in range(0, features.shape[0]):
        if (features[i] == np.zeros(features[i].shape)).all():
            pass
        else:
            features[i] /= np.linalg.norm(features[i], ord=2)

    features[features > 1] = 1
    features[features < -1] = -1

    return features
