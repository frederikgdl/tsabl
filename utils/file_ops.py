import re
import os
import pickle
from functools import reduce
from json import dumps
from os.path import join, isfile
import numpy as np


# Read and strip lines in file
def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


# Read and strip lines in file
def read_lines(file_path):
    return read_file(file_path).splitlines()


# Read tweets from tsv file
def read_tweets_tsv_file(file_path, index=1):
    tweets = []
    with open(file_path) as f:
        for line in f:
            tweets.append(line.split('\t')[index])

    return tweets


def write_to_file(data_string, file_path):
    """
    Generic write function that writes a string to a given file.
    Will create necessary parent directories if they don't exist already.
    Will overwrite file if it exists already.

    :param data_string: The string to write
    :param file_path:   The path to the output file
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w+') as f:
        f.write(data_string)


# Saves the text of each tweet on separate lines in a new file
def write_tweets(tweets, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w+') as f:
        for tweet in tweets:
            f.write(str(tweet) + os.linesep)


def read_twitter_id_file(file_path, labeled=False):
    """
    Return a list of id strings from text file
    :param file_path: The path of the input file
    :param labeled: Set to true if ids have an annotation associated on each line
    :return: A list of id strings
    """
    ids = []
    with open(file_path) as f:
        for line in f:
            ids.append(line.strip().split()[0] if labeled else line.strip())
    return ids


def read_labeled_file(file_path):
    data = []
    labels = []

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip().split()
            data.append(' '.join(line[:-1]))
            labels.append(line[-1])

    return data, labels


def get_files_in_dir_with_extension(dir_path, extension=""):
    return [join(dir_path, f) for f in os.listdir(dir_path) if
            isfile(join(dir_path, f)) and join(dir_path, f).endswith(extension)]


# Concatenate files and return lines as strings in an array
def read_and_concatenate_files(dir_path, extension=""):
    files = get_files_in_dir_with_extension(dir_path, extension)
    return reduce(lambda a, b: a + b, [[line.strip() for line in open(f)] for f in files])


# Returns the text from a tweet object
def get_text(tweet):
    return re.sub('\s+', ' ', tweet["text"])


# Saves every JSON tweet object on a separate line in a new file
def save_json_tweets(tweets, file_path, annotations=None):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w+') as f:
        for tweet in tweets:
            f.write(dumps(tweet) + (" " + annotations[tweet["id_str"]] if annotations is not None else '') + os.linesep)


# Saves the text of each tweet on separate lines in a new file
def save_text_tweets(tweets, file_path, annotations=None):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w+') as f:
        for tweet in tweets:
            f.write(get_text(tweet) +
                    (" " + annotations[tweet["id_str"]] if annotations is not None else '') + os.linesep)


def load_word_embeddings(file_name):
    with open(file_name) as f:
        # Get dimensionality from length of first vector
        first_line = f.readline()
        dimensionality = len(first_line.split()) - 1
        embeddings = {}

        # Add the first line to embeddings
        split_first_line = first_line.split(None, 1)
        embeddings[split_first_line[0]] = np.fromstring(split_first_line[1], dtype=float, sep=' ')

        # Add the rest of the vectors
        for i, line in enumerate(f):
            split_line = line.split(None, 1)
            word = split_line[0]

            # Only add lines if they have same length as first
            if len(split_line[1].split()) == dimensionality:
                embeddings[word] = np.fromstring(split_line[1], dtype=float, sep=' ')

    return embeddings


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
                split_line = line.strip().split()
                tweet = ' '.join(split_line[:-1])
                label = split_line[-1]

            # Validate label
            if label not in valid_labels:
                print(file_name, "has invalid label: \"" + label + "\" on line", len(tweets) + 1)
                raise ValueError

            tweets.append(tweet)
            labels.append(label)

    return tweets, labels


def save_model(model, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb+") as f:
        pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)


def load_model(file_name):
    with open(file_name, "rb") as f:
        data = pickle.load(f)
    return data


def dump_embed_file(output_file, inverse_vocab_map, embeddings):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w+') as f:
        for k in inverse_vocab_map.keys():
            f.write(str(inverse_vocab_map[k]) + ' ' + ' '.join([str(num) for num in embeddings[k]]) + '\n')
