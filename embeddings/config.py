import os.path

SEED = 1234


# Embedding options

EMBEDDING_LENGTH = 50

WINDOW_SIZE = 3

HIDDEN_SIZE = 20

ADAGRAD_LR = 0.1


# Tokenizer options

MIN_WORD_FREQUENCY = 0

MAX_NUMBER_WORDS = None

LOWERCASE = True


# File options

DATA_FILE = os.path.abspath("data/filtered/tweets.txt")
DATA_FILE_LABELED = True

VOCAB_FILE = os.path.abspath("data/raw/vocab.txt")
