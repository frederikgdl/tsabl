#!/usr/bin/env python
import logging
from os import path, listdir

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

import classifiers.train_and_test as train_and_test
from classifiers.models.lexicon_classifier import LexiconClassifier
from classifiers.models.log_res import LogRes
from classifiers.models.svm import SVM

import scripts.config as config


# Directory containing embeddings of different epochs
embeddings_dir = path.join(config.EMBEDDINGS_DIR, config.SELECTED_EMBEDDINGS)
embeddings_files = []

# Directory to store results
# Store results for each epoch
results_dir = path.join(config.RESULT_DIR, config.SELECTED_EMBEDDINGS)

logger = None
verbose = 0
quiet = False


# Classifiers to use are defined in this function.
# By having this in a function, we know that fresh instances are trained and tested every epoch.
def classifiers():
    return [SVM(name="SVM c=1", c=1), LogRes(), LexiconClassifier()]


# The metrics to graph. The keys must match the keys of Model.Result. The values are pretty labels.
metrics = {'ternary_macro_f1_score': 'Macro F1', 'f1_pn_score': 'F1 PN'}

# Results per classifier, per epoch. Key is name of classifier. Value is list of results.
results = {}


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
    colors = ['r', 'g', 'b', 'k']
    line_styles = ['-', '--', '.-', '---']

    cls = classifiers()

    plot_lines = []
    for i, classifier in enumerate(cls):
        color = colors[i % len(colors)]
        for j, metric in enumerate(metrics.keys()):
            line_style = line_styles[j % len(line_styles)]
            line = plt.plot(x,
                            list(map(lambda r: r[metric], results[classifier.name])),
                            line_style,
                            color=color)
            plot_lines.append(line)

    # plt.axis([0, len(embeddings_files), 0, 1])
    plt.xticks(x)
    plt.xlabel('Epoch')
    plt.ylabel('Scores')

    # Classifier legend (colors)
    color_patches = []
    for i, classifier in enumerate(cls):
        color = colors[i % len(colors)]
        patch = mpatches.Patch(color=color, label=classifier.name)
        color_patches.append(patch)

    classifier_legend = plt.legend(handles=color_patches, loc='upper right')

    # Metric legend (line styles)
    line_style_handles = []
    for i, metric in enumerate(metrics.keys()):
        line_style = line_styles[i % len(line_styles)]
        line = mlines.Line2D([], [], color='k', linestyle=line_style, label=metrics[metric])
        line_style_handles.append(line)

    plt.legend(handles=line_style_handles, loc='lower right')
    plt.gca().add_artist(classifier_legend)

    plt.savefig(path.join(results_dir, 'figure.png'))
    plt.show()


def main():
    setup_logger()

    # Sort by suffix number of files. Turn to int so that '7' is treated as less that '18', for instance.
    global embeddings_files
    embeddings_files = sorted(listdir(embeddings_dir), key=lambda f: int(f.split("-")[-1]))

    for embeddings_file in embeddings_files:
        logger.info(embeddings_file)

        # Configure test_and_train
        train_and_test.classifiers = classifiers()
        train_and_test.baselines = []
        train_and_test.embedding_file = path.join(embeddings_dir, embeddings_file)
        train_and_test.verbose = -2
        train_and_test.quiet = True
        train_and_test.results_dir = path.join(results_dir, embeddings_file)

        # Run
        train_and_test.main()

        # Save results
        for classifier in train_and_test.classifiers:
            result = classifier.results.clone()
            if classifier.name not in results:
                results[classifier.name] = []
            results[classifier.name].append(result)

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
