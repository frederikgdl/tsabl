import os
import re
import pickle

import numpy as np
from sklearn.preprocessing import scale


# Reduce letters that occurs more than three times to max three
def reduce_lengthening(text):
    pattern = re.compile(r"(.)\1{2,}")
    return pattern.sub(r"\1\1\1", text)


# Load tweets into list of list of words
def load_tweets(file_name):
    tweets = []

    with open(file_name) as f:
        for line in f:
            line = line.strip().split(' ')

            tweets.append(line)

    return tweets


# Load tweets into list of strings
def load_tweets_full(file_name):
    tweets = []

    with open(file_name) as f:
        for tweet in f:
            tweet = tweet.strip('\n')
            tweets.append(tweet)

    return tweets


def save_tweets(tweets, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w+') as f:
        for tweet in tweets:
            f.write(tweet + '\n')


def load_labeled_data(file_name):
    """
    Load a tab separated file with tweet in first column and label in second column
    :param file_name: The tab separated file to load
    :return: A two-element tuple of original tweets and labels
    """
    tweets = []
    labels = []

    valid_labels = ["positive", "negative", "neutral"]

    file_extension = os.path.splitext(file_name)[1]

    with open(file_name) as f:
        for line in f:

            # If .tsv, assume tweet and label are separated by a tab character, and that there is only one tab per line.
            if file_extension == ".tsv":
                split_line = line.strip().split('\t')
                if len(split_line) != 2:
                    print(file_name, "contained line with" + str(len(split_line) - 1), "tabs, not 1.")
                    raise ValueError

                tweet = split_line[0]
                label = split_line[1]

            # If not .tsv, assume text file where last word in line is label
            else:
                split_line = line.strip().split(" ")
                tweet = ' '.join(split_line[:-1])
                label = split_line[-1]

            # Validate label
            if label not in valid_labels:
                print(file_name, "has invalid label: \"" + label + "\" on line", len(tweets) + 1)
                raise ValueError

            tweets.append(tweet)
            labels.append(label)

    return tweets, labels


def load_word_embeddings(file_name):
    with open(file_name) as f:
        dimensionality = int(f.readline().split()[1])
        embeddings = {}

        for i, line in enumerate(f):
            split_line = line.strip().split(' ', 1)
            word = split_line[0]

            # GloVe had a space as char, first number used instead
            # Plus erroneous files
            if len(split_line[1].split(' ')) == dimensionality:
                embeddings[word] = np.fromstring(split_line[1], dtype=float, sep=' ')

    return embeddings


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


def save_model(model, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb+") as f:
        pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)


def load_model(file_name):
    with open(file_name, "rb") as f:
        data = pickle.load(f)
    return data


def remove_neutrals(predictions_num, rel_docs):
    return np.array([rel_docs if pred == rel_docs else rel_docs * -1 for pred in predictions_num])
