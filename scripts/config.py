from os import path

RESULT_DIR = path.abspath('results/')

EMBEDDINGS_DIR = path.abspath('data/embeddings/')

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
    "ComboB"
    "EMOTICON.150K",
    "EMOTICON_EXT",
    "LexiconClassifier",
    "TEXTBLOB",
    "VADER",
]

# Skip if number of epoch files is less than this number
NUM_EPOCHS = 30
