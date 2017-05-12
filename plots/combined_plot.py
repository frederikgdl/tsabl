from os import path, environ

import matplotlib
import matplotlib.patches as mpatches

# Fix for running this script on a server without graphics.
# This line must run before importing pyplot!
from plots.results_data import ResultsData

if 'DISPLAY' not in environ:
    matplotlib.use('Agg')

import matplotlib.pyplot as plt

from plots import config

name_of_classifier = config.CLASSIFIER.lower()
metric = config.METRIC[0]
metric_pretty = config.METRIC[1]

num_epochs = config.NUM_EPOCHS

number_of_figures = len(config.METHODS)
fig_number = 1

# The global variable 'data' will be set by the main function
data = None


# Plot
def plot(method):
    x = list(range(1, 1 + num_epochs))
    colors = config.COLORS
    line_style = '-'

    global fig_number
    fig = plt.figure(fig_number, (10, 6))
    fig_number += 1
    fig_title = method + " " + config.EMBEDDINGS_KEY + " " + metric_pretty
    fig.canvas.set_window_title(fig_title)

    ax = plt.subplot(111)

    plot_lines = []

    embeddings = sorted(data[method].keys())
    for i, embedding in enumerate(embeddings):
        # Each dataset gets a color
        color = colors[i % len(colors)]
        epoch_results = data[method][embedding][name_of_classifier]
        f1_scores = list(map(lambda e: e[metric], epoch_results))
        line = ax.plot(x,
                       f1_scores,
                       line_style,
                       color=color)
        plot_lines.append(line)

    # plt.axis([0, len(embeddings_files), 0, 1])
    plt.xticks(x)
    plt.xlabel('Epoch')
    plt.ylabel('Score')

    # Classifier legend (colors)
    color_patches = []
    for i, embedding in enumerate(embeddings):
        color = colors[i % len(colors)]
        patch = mpatches.Patch(color=color, label=embedding)
        color_patches.append(patch)

    # Show only every second epoch label
    for label in ax.xaxis.get_ticklabels()[::2]:
        label.set_visible(False)

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    ax.legend(handles=color_patches, loc='upper right', bbox_to_anchor=(1.3, 1.0))
    save_path = path.join(config.RESULT_DIR, method, fig_title.replace(' ', '_'))
    print("Saving to", save_path)
    plt.savefig(save_path)
    plt.show(block=fig_number == number_of_figures + 1)


def main():
    global data
    data = ResultsData(config.METHODS, config.EMBEDDINGS, [name_of_classifier], num_epochs)
    for method in data.methods:
        plot(method)


if __name__ == "__main__":
    main()
