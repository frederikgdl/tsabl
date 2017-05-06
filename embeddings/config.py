from os import path

SEED = 1234


# Embedding options

EPOCHS = 30

MARGIN = 1

BATCH_SIZE = 30

EMBEDDING_LENGTH = 50

WINDOW_SIZE = 7

HIDDEN_SIZE = 20

LEARNING_RATE = 0.1

DROPOUT_P = 0

ALPHA = 0.5

SENTIMENT_CLASSES = 3

USE_ADAGRAD = False


# Tokenizer options

# int or None
MIN_WORD_FREQUENCY = 5

# int or None
MAX_NUMBER_WORDS = None

LOWERCASE = True


# File options

OUTPUT_FILE = path.abspath('data/embeddings/test_embeddings.txt')

POS_DATA_FILE = path.abspath('data/filtered/emoticon.pos.0.txt')

NEG_DATA_FILE = path.abspath('data/filtered/emoticon.neg.0.txt')

NEU_DATA_FILE = path.abspath('data/filtered/emoticon.neu.0.txt')

# POS_DATA_FILE = path.abspath('data/preprocessed/datasets/1M/tweets.AFINN.pos.txt')
#
# NEG_DATA_FILE = path.abspath('data/preprocessed/datasets/1M/tweets.AFINN.neg.txt')
#
# NEU_DATA_FILE = path.abspath('data/preprocessed/datasets/1M/tweets.AFINN.neu.txt')
