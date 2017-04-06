from os import path

# File options
EMBEDDING_FILE = path.abspath("data/embeddings/binary_sa_embedding/afinn-1M-round-19")
TRAIN_FILE = path.abspath("data/preprocessed/twitter-2016train-A.txt")
TEST_FILE = path.abspath("data/preprocessed/twitter-2016devtest-A.txt")

SVM_MODEL_FILE = path.abspath("models/svm.pickle")
LOGRES_MODEL_FILE = path.abspath("models/logres.pickle")
