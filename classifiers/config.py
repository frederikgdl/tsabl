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

SUFFIX = "agg_ternary/LexiconClassifier/embeddings-100-LexiconClassifier-round-19"

# File options
EMBEDDING_FILE = path.abspath("data/embeddings/" + SUFFIX)
# EMBEDDING_FILE = path.abspath("data/embeddings/embeddings.txt")
TRAIN_FILE = path.abspath("data/preprocessed/twitter-2013train-A.txt")
TEST_FILE = path.abspath("data/preprocessed/twitter-2013dev-A.txt")

# K-fold validation
# Set K to a number greater than 1 to activate K-fold validation.
# The TRAIN_FILE will be split into K number of chunks, where 1 chunk is used for testing and the rest for training.
# All chunks will be used as test set by turn. This means training will happen K times, and
# the average score is returned.
K = -1

SKIP_TRAINING = False
SKIP_TESTING = False

MODELS_DIR = path.abspath("models")
RESULTS_DIR = path.abspath("results/svm-tests/" + SUFFIX)

CLASSIFIERS = [
    SVM(name="SVM c=1e-3 dual=False", c=1e-3, dual=False),
    SVM(name="SVM c=1e-2 dual=False", c=1e-2, dual=False),
    SVM(name="SVM c=1e-1 dual=False", c=1e-1, dual=False),
    SVM(name="SVM c=1 dual=False", c=1, dual=False),
    SVM(name="SVM c=1e1 dual=False", c=1e1, dual=False),
    SVM(name="SVM c=1e2 dual=False", c=1e2, dual=False),
    SVM(name="SVM c=1e3 dual=False", c=1e3, dual=False),
    SVM(name="SVM c=2e-1 dual=False", c=2e-1, dual=False),
    SVM(name="SVM c=3e-1 dual=False", c=3e-1, dual=False),
    SVM(name="SVM c=4e-1 dual=False", c=4e-1, dual=False),
    SVM(name="SVM c=5e-1 dual=False", c=5e-1, dual=False),
    SVM(name="SVM c=6e-1 dual=False", c=6e-1, dual=False),
    SVM(name="SVM c=7e-1 dual=False", c=7e-1, dual=False),
    SVM(name="SVM c=8e-1 dual=False", c=8e-1, dual=False),
    SVM(name="SVM c=9e-1 dual=False", c=9e-1, dual=False),
    SVM(name="SVM c=2e-2 dual=False", c=2e-2, dual=False),
    SVM(name="SVM c=3e-2 dual=False", c=3e-2, dual=False),
    SVM(name="SVM c=4e-2 dual=False", c=4e-2, dual=False),
    SVM(name="SVM c=5e-2 dual=False", c=5e-2, dual=False),
    SVM(name="SVM c=6e-2 dual=False", c=6e-2, dual=False),
    SVM(name="SVM c=7e-2 dual=False", c=7e-2, dual=False),
    SVM(name="SVM c=8e-2 dual=False", c=8e-2, dual=False),
    SVM(name="SVM c=9e-2 dual=False", c=9e-2, dual=False),
    SVM(name="SVM c=1.1 dual=False", c=1.1, dual=False),
    SVM(name="SVM c=1.2 dual=False", c=1.2, dual=False),
    SVM(name="SVM c=1.3 dual=False", c=1.3, dual=False),
    SVM(name="SVM c=1.4 dual=False", c=1.4, dual=False),
    SVM(name="SVM c=1.5 dual=False", c=1.5, dual=False),
    SVM(name="SVM c=1.6 dual=False", c=1.6, dual=False),
    SVM(name="SVM c=1.7 dual=False", c=1.7, dual=False),
    SVM(name="SVM c=1.8 dual=False", c=1.8, dual=False),
    SVM(name="SVM c=1.9 dual=False", c=1.9, dual=False),
    SVM(name="SVM c=2.0 dual=False", c=2.0, dual=False),
    # SVM(name="SVM c=1 dual=False", c=2e-2, dual=False),
    # SVM(name="SVM c=1 dual=True", c=2e-2, dual=True),
    # SVM(name="SVM c=2e-2", c=2e-2),
    # SVM(name="SVM c=1", c=1),
    # SVM(name="SVM c=1e-3", c=1e-3),
    # SVM(name="SVM c=1e-2", c=1e-2),
    # SVM(name="SVM c=2e-2", c=2e-2),
    # SVM(name="SVM c=3e-2", c=3e-2),
    # SVM(name="SVM c=4e-2", c=4e-2),
    # SVM(name="SVM c=5e-2", c=5e-2),
    # SVM(name="SVM c=6e-2", c=6e-2),
    # SVM(name="SVM c=7e-2", c=7e-2),
    # SVM(name="SVM c=8e-2", c=8e-2),
    # SVM(name="SVM c=9e-2", c=9e-2),
    # SVM(name="SVM c=2e-1", c=2e-1),
    # SVM(name="SVM c=3e-1", c=3e-1),
    # SVM(name="SVM c=4e-1", c=4e-1),
    # SVM(name="SVM c=5e-1", c=5e-1),
    # SVM(name="SVM c=6e-1", c=6e-1),
    # SVM(name="SVM c=7e-1", c=7e-1),
    # SVM(name="SVM c=8e-1", c=8e-1),
    # SVM(name="SVM c=9e-1", c=9e-1),
    # SVM(name="SVM c=1e-1", c=1e-1),
    # SVM(name="SVM c=1e1", c=1e1),
    # SVM(name="SVM c=1e2", c=1e2),
    # SVM(name="SVM c=1e3", c=1e3),
    # LogRes()
]


BASELINES = [
    RandomUniform(),
    RandomWeighted(),
    AfinnModel(),
    Vader(threshold=0.1),
    Textblob(subjectivity_threshold=0.1, polarity_threshold=0.5),
    ComboAverage(name="ComboA", a=0, b=4, c=4, d=2),
    ComboAverage(name="ComboB", a=3, b=1, c=1, d=1),
    ComboAverage(),
    LexiconClassifier()
]

VERBOSE = 0
QUIET = False
