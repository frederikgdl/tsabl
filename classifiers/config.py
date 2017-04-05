from os import path

# File options
EMBEDDING_FILE = path.abspath("data/embeddings/embeddings.txt")
TRAIN_FILE = path.abspath("data/raw/twitter-2013train-A.txt")
TEST_FILE = path.abspath("data/raw/twitter-2013test-A.txt")

SVM_MODEL_FILE = path.abspath("models/svm.pickle")
LOGRES_MODEL_FILE = path.abspath("models/logres.pickle")
