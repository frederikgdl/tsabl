import random

from utils import file_ops


# Returns tweets and labels for tweets in files in file_paths
# Parameter sentiment_labels contains labels for all tweets in file_paths
# All tweets in the file given by the first file path is labeled by the first label in sentiment_labels
def get_training_data(file_paths, sentiment_labels):
    assert len(file_paths) == len(sentiment_labels)

    tweets, labels = [], []
    for i, file_path in enumerate(file_paths):
        new_tweets = file_ops.read_lines(file_path)
        new_labels = [sentiment_labels[i]]*len(new_tweets)

        tweets += new_tweets
        labels += new_labels

    return tweets, labels


def shuffle_data(texts, labels):
    zipped = list(zip(texts, labels))
    random.shuffle(zipped)
    unzipped = [list(t) for t in zip(*zipped)]
    return unzipped[0], unzipped[1]


def get_numeric_labels(labels, length=2):
    if length == 2:
        return [[1, -1] if label == 'positive' else [-1, 1] for label in labels]
    elif length == 3:
        return [[1, -1, -1] if label == 'positive'
                else [-1, -1, 1] if label == 'negative'
                else [-1, 1, -1] for label in labels]
    else:
        return None


def get_context_windows_labels(text_sequences, labels, window_size):
    # TODO: padding?
    """
    Create context windows and labels lists from text sequences
    Ignores texts (tweets) shorter than the window size
    :param text_sequences: list of text sequences (vectors with word indices)
    :param labels: list of labels (ints)
    :param window_size: int with size of context windows
    :return: context_windows: list with context windows, each windows a list of size window_size
             new_labels:
    """
    context_windows = []
    new_labels = []
    for idx, seq in enumerate(text_sequences):
        if len(seq) < window_size:
            continue

        for i in range(len(seq) - window_size + 1):
            window = []
            for j in range(window_size):
                window.append(seq[i + j])

            context_windows.append(window)
            new_labels.append(labels[idx])

    return context_windows, new_labels


def get_negative_samples(context_windows, vocab_size):
    middle_idx = len(context_windows[0])//2
    negative_samples = []
    for window in context_windows:
        neg_sample = list(window)
        rand_idx = random.randint(1, vocab_size-1)

        while rand_idx == middle_idx:
            rand_idx = random.randint(1, vocab_size-1)

        neg_sample[middle_idx] = rand_idx
        negative_samples.append(neg_sample)

    return negative_samples


def dump_embed_file(output_file, inverse_vocab_map, embeddings):
    file_ops.dump_embed_file(output_file=output_file, inverse_vocab_map=inverse_vocab_map, embeddings=embeddings)
