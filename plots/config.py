from collections import OrderedDict
from os import path

from classifiers.models.lexicon_classifier import LexiconClassifier
from classifiers.models.random_uniform import RandomUniform
from classifiers.models.svm import SVM

RESULT_DIR = path.abspath('results/dev')

EMBEDDINGS_DIR = path.abspath('data/embeddings/')


# Classifiers to use are defined in this function.
# By having this in a function, we know that fresh instances are trained and tested every epoch.
def classifiers():
    return [SVM(name="SVM c=6e-3 dual=False", c=6e-3, dual=False), RandomUniform(), LexiconClassifier()]

# The metrics to graph. The keys must match the keys of Model.Result. The values are pretty labels.
METRICS = OrderedDict()
METRICS['ternary_macro_f1_score'] = 'Macro F1'
METRICS['f1_pn_score'] = 'F1 PN'

##########################################################################

# test_all_datasets
METHODS = [
    #"binary",
    #"ternary",
    "agg_ternary",
]

embedding_sets = {
    "datasets": [
        "AFINN",
        "ComboA",
        "ComboB",
        "EMOTICON.150K",
        "EMOTICON_EXT",
        "LexiconClassifier",
        "TEXTBLOB",
        "VADER",
    ],
    "alpha": [
        "alpha-0.0",
        "alpha-0.1",
        "alpha-0.2",
        "alpha-0.4",
        "alpha-0.5",
        "alpha-0.6",
        "alpha-0.8",
        "alpha-1.0",
    ],
    "windowsize": [
        "windowsize-3",
        "windowsize-5",
        "windowsize-7",
        "windowsize-9",
    ],
    "learningrate": [
        "learningrate-0.001",
        "learningrate-0.01",
        "learningrate-0.05",
        "learningrate-0.2",
        "learningrate-0.3",
        "learningrate-0.5",
        "learningrate-0.7",
        "learningrate-0.9",
        "learningrate-1.1",
    ],
    "margin": [
        "margin-0.5",
        "margin-0.7",
        "margin-0.9",
        "margin-1.0",
        "margin-1.1",
        "margin-1.3",
        "margin-1.5",
        "margin-1.7",
        "margin-1.9",
        "margin-2.0",
        "margin-3.0",
        "margin-4.0",
        "margin-5.0",
        "margin-10.0",
    ],
    "embeddinglen": [
        "embeddinglen-50",
        "embeddinglen-75",
        "embeddinglen-100",
        "embeddinglen-125",
        "embeddinglen-150",
    ],
    "hiddenlength": [
        "hiddenlength-10",
        "hiddenlength-20",
        "hiddenlength-30",
        "hiddenlength-50",
        "hiddenlength-100",
    ]
}

pretty = {
    "alpha": "Alpha",
    "windowsize": "Window Size",
    "learningrate": "Learning Rate",
    "margin": "Margin",
    "embeddinglen": "Embedding Length",
    "hiddenlength": "Hidden Layer Size",
    "AFINN": "AFINN",
    "ComboA": "Combo A",
    "ComboB": "Combo B",
    "EMOTICON.150K": "Emoticon",
    "EMOTICON_EXT": "Emoticon Ext.",
    "LexiconClassifier": "FJLC",
    "TEXTBLOB": "TextBlob",
    "VADER": "VADER",
}

# Embeddings/datasets to train, test and plot.
# If set to the string 'all', all subdirectories of <method> will be checked,
# which might include more than those listed above.
EMBEDDINGS_KEY = "datasets"

if EMBEDDINGS_KEY != "all":
    EMBEDDINGS = embedding_sets[EMBEDDINGS_KEY]
else:
    EMBEDDINGS = "all"

# Skip if number of epoch files is less than this number
NUM_EPOCHS = 20


##########################################################################
# PLOTTING STUFF

COLORS = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', '#ffa500']
LINE_STYLES = ['-', '--', '.-', '---']

# combined_plot.py and combined_tables.py
METRIC = list(METRICS.items())[0]
CLASSIFIER = "svm c=1"
