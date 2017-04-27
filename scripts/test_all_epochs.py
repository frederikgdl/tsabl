#!/usr/bin/env python
import classifiers.train_and_test as train_and_test
from classifiers.models.svm import SVM
from os import path, listdir
import matplotlib.pyplot as plt
import logging

path_of_this_file = path.dirname(path.realpath(__file__))

# Directory containing embeddings of different epochs
embeddings_dir = path.join(path_of_this_file, "../data/embeddings/binary_sa_embedding")
embeddings_files = []

logger = None
verbose = 0
quiet = False

results = []


def setup_logger():
    levels = [logging.ERROR, logging.WARNING, logging.INFO, logging.DEBUG]
    level = levels[min(len(levels) - 1, verbose + 2)]  # capped to number of levels

    # create logger
    global logger
    logger = logging.getLogger('test_all_epochs')
    logger.setLevel(level)
    logger.propagate = False

    if quiet:
        logging.disable(logging.ERROR)
        return logger

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # create formatter
    formatter = logging.Formatter("%(asctime)s\t%(levelname)s\t%(message)s")

    # add formatter to ch
    ch.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(ch)


# Plot
def plot():
    x = list(range(1, 1 + len(embeddings_files)))
    plt.plot(x, list(map(lambda r: r['ternary_macro_f1_score'], results)), 'b-', label='Macro F1')
    plt.plot(x, list(map(lambda r: r['f1_pn_score'], results)), 'r--', label='F1 PN')
    # plt.axis([0, len(embeddings_files), 0, 1])
    plt.xticks(x)
    plt.xlabel('Epoch')
    plt.ylabel('Scores')
    plt.legend()
    plt.show()


def main():
    setup_logger()

    # Sort by suffix number of files. Turn to int so that '7' is treated as less that '18', for instance.
    global embeddings_files
    embeddings_files = sorted(listdir(embeddings_dir), key=lambda f: int(f.split("-")[-1]))

    for embeddings_file in embeddings_files:
        logger.info(embeddings_file)

        # Configure test_and_train
        train_and_test.classifiers = [SVM()]
        train_and_test.baselines = []
        train_and_test.embedding_file = path.join(embeddings_dir, embeddings_file)
        train_and_test.verbose = -2
        train_and_test.quiet = True

        # Run
        train_and_test.main()

        # Save results
        result = train_and_test.classifiers[0].results.clone()
        results.append(result)

    plot()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='test SVM with word embeddings of different epochs and plot a graph')

    parser.add_argument('-e', '--embeddings-dir', default=embeddings_dir,
                        help='Directory containing word embeddings. Each file should end with number of epoch.')

    # Logger verbosity parameters
    vgroup = parser.add_argument_group('verbosity arguments')
    vgroup.add_argument('-v', '--verbose', action='count', default=0, help='verbosity level, repeat to increase')
    vgroup.add_argument('-q', '--quiet', action='store_true', help='no print to console')

    args = parser.parse_args()

    embeddings_dir = args.embeddings_dir
    verbose = args.verbose
    quiet = args.quiet

    main()
