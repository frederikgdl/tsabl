import os.path

SEED = 1234


# Embedding options

EPOCHS = 3

MARGIN = 1

BATCH_SIZE = 32

EMBEDDING_LENGTH = 50

WINDOW_SIZE = 7

HIDDEN_SIZE = 20

ADAGRAD_LR = 0.1

DROPOUT_P = 0


# Tokenizer options

# int or None
MIN_WORD_FREQUENCY = 5

# int or None
MAX_NUMBER_WORDS = None

LOWERCASE = True


# File options

DATA_FILE = os.path.abspath("data/filtered/tweets.txt")
DATA_FILE_LABELED = True

VOCAB_FILE = os.path.abspath("data/raw/vocab.txt")

OUTPUT_FILE = os.path.abspath("data/embeddings/embeddings.txt")
