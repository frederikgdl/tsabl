from os import path

from classifiers.models.lexicon_classifier import LexiconClassifier
from classifiers.models.random_uniform import RandomUniform
from classifiers.models.svm import SVM

RESULT_DIR = path.abspath('results/')

EMBEDDINGS_DIR = path.abspath('data/embeddings/')


# Classifiers to use are defined in this function.
# By having this in a function, we know that fresh instances are trained and tested every epoch.
def classifiers():
    return [SVM(name="SVM c=1", c=1), RandomUniform(), LexiconClassifier()]

# The metrics to graph. The keys must match the keys of Model.Result. The values are pretty labels.
METRICS = {
    'ternary_macro_f1_score': 'Macro F1',
    'f1_pn_score': 'F1 PN',
}

##########################################################################

# test_all_epochs:
# Embeddings to test
# This is the sub-folder of the directory data/embeddings/ that contain embeddings with different epochs
# Embedding files end on number indicating epoch round
# Results are stored in similar path under the directory results/
SELECTED_EMBEDDINGS = 'binary/VADER/'

##########################################################################

# test_all_datasets
METHODS = [
    "binary",
    "ternary",
    "agg_ternary",
]

EMBEDDINGS = [
    "AFINN",
    "ComboA",
    "ComboB",
    "EMOTICON.150K",
    "EMOTICON_EXT",
    "LexiconClassifier",
    "TEXTBLOB",
    "VADER",
]

# Skip if number of epoch files is less than this number
NUM_EPOCHS = 30
