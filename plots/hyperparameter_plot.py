from os import path, environ, listdir

import matplotlib
import numpy as np

from plots.results_data import ResultsData
# Fix for running this script on a server without graphics.
# This line must run before importing pyplot!
from utils.misc import sorted_by_suffix

if 'DISPLAY' not in environ:
    matplotlib.use('Agg')

import matplotlib.pyplot as plt

from plots import config

HYPERPARAMETERS = {'alpha': 'Alpha',
                   'embeddinglen': 'Word embedding length',
                   'hiddenlength': 'Size of hidden layers',
                   'learningrate': 'Learning rate',
                   'margin': 'Margin',
                   'windowsize': 'Size of context window'}

METHOD = 'agg_ternary'
RESULTS_DIR = path.join(config.RESULT_DIR, METHOD)
CLASSIFIER = 'SVM c=1'
METRIC = 'ternary_macro_f1_score'
METRIC_PRETTY = 'Macro F1'
EPOCH_START = 10
EPOCH_STOP = 20
NUM_EPOCHS = config.NUM_EPOCHS

number_of_figures = len(HYPERPARAMETERS)
fig_number = 1


def get_dir_names(prefix):
    return sorted_by_suffix(
        [d for d in listdir(RESULTS_DIR) if path.isdir(path.join(RESULTS_DIR, d)) and d.startswith(prefix)])


def get_avg_from_epochs(data, method, embedding, classifier, metric, epoch_start, epoch_stop):
    assert epoch_start <= epoch_stop
    epoch_results = data[method][embedding][classifier][epoch_start - 1:epoch_stop]
    metric_scores = list(map(lambda e: e[metric], epoch_results))
    return sum(metric_scores) / len(metric_scores)


def plot(hyperparameter):
    color = 'blue'
    line_style = 'o-'

    dir_names = get_dir_names(hyperparameter)

    hyperparameter_values = [dir_name.split("-")[-1] for dir_name in dir_names]

    data = ResultsData([METHOD], dir_names, [CLASSIFIER], NUM_EPOCHS)

    avg_scores = [get_avg_from_epochs(data, METHOD, dir_name, CLASSIFIER, METRIC,
                                      epoch_start=EPOCH_START, epoch_stop=EPOCH_STOP) for dir_name in dir_names]

    global fig_number
    fig = plt.figure(fig_number, (10, 6))
    fig_number += 1
    fig_title = 'comparison ' + hyperparameter
    fig.canvas.set_window_title(fig_title)

    ax = plt.subplot(111)

    a = np.arange(len(hyperparameter_values))
    ax.plot(a, avg_scores, line_style, color=color)

    ax.xaxis.set_ticks(a)
    ax.xaxis.set_ticklabels(hyperparameter_values)
    #plt.xticks(hyperparameter_values, rotation=30)
    plt.xlabel(HYPERPARAMETERS[hyperparameter])
    plt.ylabel(METRIC_PRETTY)
    plt.tight_layout()

    plt.savefig(path.join(config.RESULT_DIR, METHOD, fig_title.replace(' ', '_') + '.png'))
    plt.show(block=fig_number == number_of_figures + 1)


def main():
    for hyperparameter in list(HYPERPARAMETERS.keys()):
        plot(hyperparameter)


if __name__ == '__main__':
    main()
