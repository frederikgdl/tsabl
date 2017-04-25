import numpy as np
from utils import file_ops


class WordEmbeddingDict:
    def __init__(self, data_file_name):
        self.embeddings = {}
        self.n_vecs = 0
        self.d = 0
        self.data_file_name = data_file_name
        self.load_embeddings()

    # Loads words and word embeddings and stores in self.words and self.embeddings
    # Decodes utf-8
    # File format:
    # word0 embedding0 embedding1 ...
    # word1 embedding0 embedding1 ...
    # ...
    def load_embeddings(self):
        self.embeddings = file_ops.load_word_embeddings(self.data_file_name)
        self.n_vecs = len(self.embeddings)
        self.d = len(self.embeddings[list(self.embeddings.keys())[0]])

    # Returns word embedding for a string word as numpy array
    # Only returns exact match for word
    # Returns None if word not exists in vocabulary
    def get_word_embedding(self, word):
        return self.embeddings.get(word)

    # Returns the vocabulary for this vocabulary as a list
    def get_vocabulary(self):
        return list(self.embeddings.keys())

    # Returns embedding for a tweet as numpy array
    # Tweet is list of strings
    # Default mode is average of vectors
    #
    # TODO Implement more sentence embedding methods
    #
    def get_tweet_embedding(self, tweet, mode="avg"):
        if isinstance(tweet, list):
            if mode == "avg":
                return self.calc_avg(tweet)
            else:
                print("Unsupported mode " + mode)
        else:
            raise ValueError('Tweet must be represented as a list of strings')

    # Returns the average vector of the word embeddings
    # for the words in the list of strings words
    # Returned value has type numpy array
    # Ignores words that has no word embedding
    def calc_avg(self, words):
        vecs = []
        for word in words:
            emb = self.get_word_embedding(word)
            if emb is not None:
                vecs.append(emb)

        if len(vecs) < 1:
            return np.zeros(self.d)

        vecs = np.array(vecs)
        return np.mean(vecs, axis=0)