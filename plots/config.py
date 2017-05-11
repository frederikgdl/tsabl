from collections import OrderedDict
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
METRICS = OrderedDict()
METRICS['ternary_macro_f1_score'] = 'Macro F1'
METRICS['f1_pn_score'] = 'F1 PN'

##########################################################################

# test_all_datasets
METHODS = [
    "binary",
    "ternary",
    "agg_ternary",
]

# Embeddings/datasets to train, test and plot.
# If set to the string 'all', all subdirectories of <method> will be checked.
EMBEDDINGS = "all"
"""
EMBEDDINGS = [
#    "alpha-0.0",
#    "alpha-0.2",
#    "alpha-0.4",
#    "alpha-0.5",
#    "alpha-0.6",
#    "alpha-0.8",
#    "alpha-1.0",
#    "windowsize-3",
#    "windowsize-5",
#    "windowsize-7",
#    "windowsize-9",
    "AFINN",
    "ComboA",
    "ComboB",
    "EMOTICON.150K",
    "EMOTICON_EXT",
    "LexiconClassifier",
    "TEXTBLOB",
    "VADER",
]
"""

# Skip if number of epoch files is less than this number
NUM_EPOCHS = 30


##########################################################################
# PLOTTING STUFF

COLORS = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', '#ffa500']
LINE_STYLES = ['-', '--', '.-', '---']

# combined_plot.py and combined_tables.py
METRIC = list(METRICS.items())[0]
CLASSIFIER = "svm c=1"
