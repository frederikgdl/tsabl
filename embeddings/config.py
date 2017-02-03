import os.path

SEED = 1234

EMBEDDING_LENGTH = 100

WINDOW_SIZE = 5

MIN_WORD_FREQUENCY = 5

HIDDEN_SIZE = 100

DATA_FILE = os.path.abspath("../data/raw/train_gold.id")
DATA_FILE_LABELED = True

VOCAB_FILE =  os.path.abspath("../data/raw/vocab.txt")
