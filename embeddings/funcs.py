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


# Special labels, are split and multiplied with predicted scores in loss function
def get_numeric_labels(labels, length=2):
    if length == 2:
        return [[1, -1] if label == 'positive' else [-1, 1] for label in labels]
    elif length == 3:
        return [[1, -1, 0, 1, 0, -1] if label == 'positive'
                else [0, -1, 1, -1, 0, 1] if label == 'negative'
                else [-1, 1, 0, 0, 1, -1] for label in labels]
    else:
        return None


def get_context_windows_labels(tweets, labels, window_size, vocab_map):
    """
    Create context windows and labels lists from text sequences
    Ignores texts (tweets) shorter than the window size
    :param tweets: list of tweets
    :param labels: list of labels (ints)
    :param window_size: int with size of context windows
    :param vocab_map: vocabulary and indexes
    :return: context_windows: list with context windows, each windows a list of size window_size
             new_labels:
    """
    context_windows = []
    new_labels = []
    for idx, tweet in enumerate(tweets):
        tweet = tweet.split(' ')

        if len(tweet) < window_size:
            continue

        # Convert tweet to list of vocab indexes
        seq = []
        for word in tweet:
            if word in vocab_map:
                seq.append(vocab_map[word])
            else:
                seq.append(vocab_map['<unk>'])

        # Create windows of specified size
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
        rand_idx = random.randint(0, vocab_size-1)

        while rand_idx == middle_idx:
            rand_idx = random.randint(0, vocab_size-1)

        neg_sample[middle_idx] = rand_idx
        negative_samples.append(neg_sample)

    return negative_samples


def dump_embed_file(output_file, inverse_vocab_map, embeddings):
    file_ops.dump_embed_file(output_file=output_file, inverse_vocab_map=inverse_vocab_map, embeddings=embeddings)


def get_vocab(tweet_texts, min_freq):
    """
    Build and return a map of word to a unique index.
    Words with frequency lower than min_freq are ignored.
    Higher frequency words are assigned lower indexes.
    The words in the map define the vocabulary given from the tweets.
    """

    # Build a word frequency map
    word_frequency_map = {}
    for tweet in tweet_texts:
        for word in tweet.split(' '):
            if word in word_frequency_map:
                word_frequency_map[word] += 1
            else:
                word_frequency_map[word] = 1

    # Now turn it around: Map every frequency to a list of words of that frequency
    frequency_map = {}
    for word in word_frequency_map:
        freq = word_frequency_map[word]
        if freq < min_freq:
            continue
        if freq not in frequency_map:
            frequency_map[freq] = []
        frequency_map[freq].append(word)

    # Now build the vocab/index map
    # Each word gets assigned an index by frequency. That is, the words with highest frequencies get assigned the
    # lowest indexes
    # Special padding words are put in the map
    vocab_map = {'<unk>': 0, '<s>': 1, '</s>': 2}
    idx = 3
    for freq in sorted(frequency_map.keys(), reverse=True):
        for word in frequency_map[freq]:
            vocab_map[word] = idx
            idx += 1

    return vocab_map


def pad_tweets(tweets):
    return ['<s> ' + tweet + ' </s>' for tweet in tweets]
