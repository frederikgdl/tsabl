from os import path

# File options
# EMBEDDING_FILE = path.abspath("data/embeddings/binary_sa_embedding/afinn-1M-round-19")
EMBEDDING_FILE = path.abspath("data/embeddings/embeddings.txt")
TRAIN_FILE = path.abspath("data/preprocessed/twitter-2016train-A.txt")
TEST_FILE = path.abspath("data/preprocessed/twitter-2016devtest-A.txt")

# K-fold validation
# Set K to a number greater than 1 to activate K-fold validation.
# The TRAIN_FILE will be split into K number of chunks, where 1 chunk is used for testing and the rest for training.
# All chunks will be used as test set by turn. This means training will happen K times, and
# the average score is returned.
K = -1

MODELS_DIR = path.abspath("models")
RESULTS_DIR = path.abspath("results")
