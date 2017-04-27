from os import path
from classifiers.models.afinn import AfinnModel
from classifiers.models.combo_average import ComboAverage
from classifiers.models.lexicon_classifier import LexiconClassifier
from classifiers.models.log_res import LogRes
from classifiers.models.random_uniform import RandomUniform
from classifiers.models.random_weighted import RandomWeighted
from classifiers.models.svm import SVM
from classifiers.models.textblob import Textblob
from classifiers.models.vader import Vader

SUFFIX = "sswe/sswe-u.txt"

# File options
EMBEDDING_FILE = path.abspath("data/embeddings/" + SUFFIX)
# EMBEDDING_FILE = path.abspath("data/embeddings/embeddings.txt")
TRAIN_FILE = path.abspath("data/preprocessed/twitter-2013train-A.txt")
TEST_FILE = path.abspath("data/preprocessed/twitter-2013test-A.txt")

# K-fold validation
# Set K to a number greater than 1 to activate K-fold validation.
# The TRAIN_FILE will be split into K number of chunks, where 1 chunk is used for testing and the rest for training.
# All chunks will be used as test set by turn. This means training will happen K times, and
# the average score is returned.
K = -1

SKIP_TRAINING = False
SKIP_TESTING = False

MODELS_DIR = path.abspath("models")
RESULTS_DIR = path.abspath("results/" + SUFFIX)

CLASSIFIERS = [
    SVM(),
    LogRes()
]

BASELINES = [
    RandomUniform(),
    RandomWeighted(),
    AfinnModel(),
    Vader(threshold=0.1),
    Textblob(subjectivity_threshold=0.1, polarity_threshold=0.5),
    ComboAverage(),
    LexiconClassifier()
]

VERBOSE = 0
QUIET = False
