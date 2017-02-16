import random

# def get_vocab(tweet_texts, min_freq):
#     """
#     Build and return a map of word to a unique index.
#     Higher frequency words are assigned lower indexes.
#     The words in the map define the vocabulary given from the tweets
#     """
#
#     # Build a word frequency map
#     word_frequency_map = {}
#     for tweet in tweet_texts:
#         for word in tweet.split():
#             if word in word_frequency_map:
#                 word_frequency_map[word] += 1
#             else:
#                 word_frequency_map[word] = 1
#
#     # Now turn it around: Map every frequency to a list of words of that frequency
#     frequency_map = {}
#     for word in word_frequency_map:
#         freq = word_frequency_map[word]
#         if freq < min_freq: continue
#         if freq not in frequency_map:
#             frequency_map[freq] = []
#         frequency_map[freq].append(word)
#
#     # Now build the vocab/index map
#     # Each word gets assigned an index by frequency. That is, the words with highest frequencies get assigned the
#     # lowest indexes
#     # Special padding words are put in the map
#     vocab_map = {'<unk>': 0, '<s>': 1, '</s>': 2}
#     idx = 3
#     for freq in sorted(frequency_map.keys(), reverse=True):
#         for word in frequency_map[freq]:
#             vocab_map[word] = idx
#             idx += 1
#
#     return vocab_map
#
#
# def fill_window(begin_idx, data, window_size, vocab_map):
#     """
#     Build and return a context window from a list of words.
#     Each word is replaced by its index in the vocab map,
#     or by the index for "<unk>" if not found.
#     """
#     word_ins = []
#     for i in range(window_size):
#         word = data[begin_idx + i]
#
#         if word in vocab_map:
#             word_ins.append(vocab_map[word])
#         else:
#             word_ins.append(vocab_map["<unk>"])
#
#     return word_ins


def get_numeric_labels(labels):
    return [1 if label == 'positive' else -1 if label == 'negative' else 0 for label in labels]
    # new_labels = []
    #
    # for label in labels:
    #     if label == 'positive':
    #         new_label = 1
    #     elif label == 'negative':
    #         new_label = -1
    #     else:
    #         new_label = 0
    #     new_labels.append(new_label)
    #
    # return new_labels


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
    with open(output_file, 'w+') as f:
        for k in inverse_vocab_map.keys():
            f.write(str(inverse_vocab_map[k]) + ' ' + ' '.join([str(num) for num in embeddings[k]]) + '\n')
