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

def get_context_windows(text_sequences, window_size):
    """
    Create context windows from text sequences
    Ignores texts (tweets) shorter than the window size
    :param text_sequences: list of text sequences (vectors with word indices)
    :param window_size: int with size of context windows
    :return: list with context windows, each windows a list of size window_size
    """
    context_windows = []
    for seq in text_sequences:
        if len(seq) < window_size:
            continue

        for i in range(len(seq) - window_size + 1):
            window = []
            for j in range(window_size):
                window.append(seq[i + j])

            context_windows.append(window)

    return context_windows


def get_negative_samples(context_windows, vocab_size):
    middle_idx = len(context_windows[0])//2
    negative_samples = []
    for window in context_windows:
        neg_sample = list(window)
        rand_idx = random.randint(1, vocab_size)

        while rand_idx == middle_idx:
            rand_idx = random.randint(1, vocab_size)

        neg_sample[middle_idx] = rand_idx
        negative_samples.append(neg_sample)

    return negative_samples
