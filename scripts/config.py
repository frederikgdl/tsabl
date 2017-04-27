from os import path


# test_all_epochs:
# Embeddings to test
# This is the sub-folder of the directory data/embeddings/ that contain embeddings with different epochs
# Embedding files end on number indicating epoch round
# Results are stored in similar path under the directory results/
SELECTED_EMBEDDINGS = 'binary/VADER/'

##########################################################################

RESULT_DIR = path.abspath('results/')

EMBEDDINGS_DIR = path.abspath('data/embeddings/')
