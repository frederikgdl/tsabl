from os import path


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


# File options

OUTPUT_FILE = path.abspath('data/embeddings/test_embeddings.txt')
POS_DATA_FILE = path.abspath('data/preprocessed/datasets/1M/tweets.LexiconClassifier.pos.txt')
NEG_DATA_FILE = path.abspath('data/preprocessed/datasets/1M/tweets.LexiconClassifier.neg.txt')
NEU_DATA_FILE = path.abspath('data/preprocessed/datasets/1M/tweets.LexiconClassifier.neu.txt')
