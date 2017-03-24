import os
from functools import reduce
from json import dumps
from os.path import join, isfile, basename, splitext
import numpy as np


# Read and strip lines in file
def read_file(file_path):
    with open(file_path, 'r') as f:
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
            line = line.strip().split(' ')
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
    return tweet["text"].replace("\n", " ")


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
        dimensionality = len(first_line.split(' ')) - 1
        embeddings = {}

        # Add the first line to embeddings
        split_first_line = first_line.split(' ', 1)
        embeddings[split_first_line[0]] = np.fromstring(split_first_line[1], dtype=float, sep=' ')

        # Add the rest of the vectors
        for i, line in enumerate(f):
            split_line = line.strip().split(' ', 1)
            word = split_line[0]

            # GloVe had a space as char, first number used instead
            # Plus erroneous files
            if len(split_line[1].split(' ')) == dimensionality:
                embeddings[word] = np.fromstring(split_line[1], dtype=float, sep=' ')

    return embeddings
