from os import path, environ

import matplotlib
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

# Fix for running this script on a server without graphics.
# This line must run before importing pyplot!
from plots.results_data import ResultsData

if 'DISPLAY' not in environ:
    matplotlib.use('Agg')

import matplotlib.pyplot as plt

from plots import config

classifiers = ["SVM c=1", 'Lexicon Classifier', 'RandomUniform']
metrics = config.METRICS
num_epochs = config.NUM_EPOCHS

# The global variable 'data' will be set by the main function
data = None

# A counter that ensures making different figures instead of overwriting one
fig_number = 1

# The total number of figs. Set by main. Used for 'holding' the last plot
number_of_figures = -1


# Plot
def plot(method, embedding):
    selected_embeddings = path.join(method, embedding)

    x = list(range(1, 1 + num_epochs))
    colors = config.COLORS
    line_styles = config.LINE_STYLES

    global fig_number
    fig = plt.figure(fig_number, (10, 6))
    fig_number += 1
    fig_title = selected_embeddings.replace('/', '_') + '.png'
    fig.canvas.set_window_title(fig_title)

    ax = plt.subplot(111)

    plot_lines = []
    for i, classifier in enumerate(classifiers):
        color = colors[i % len(colors)]
        for j, metric in enumerate(metrics.keys()):
            line_style = line_styles[j % len(line_styles)]
            epoch_results = data[method][embedding][classifier]
            metric_scores = list(map(lambda e: e[metric], epoch_results))
            line = ax.plot(x,
                           metric_scores,
                           line_style,
                           color=color)
            plot_lines.append(line)

    # plt.axis([0, len(embeddings_files), 0, 1])
    plt.xticks(x)
    plt.xlabel('Epoch')
    plt.ylabel('Score')

    # Classifier legend (colors)
    color_patches = []
    for i, classifier in enumerate(classifiers):
        color = colors[i % len(colors)]
        patch = mpatches.Patch(color=color, label=classifier)
        color_patches.append(patch)

    classifier_legend = plt.legend(handles=color_patches, loc='upper right')

    # Metric legend (line styles)
    line_style_handles = []
    for i, metric in enumerate(metrics.keys()):
        line_style = line_styles[i % len(line_styles)]
        line = mlines.Line2D([], [], color='k', linestyle=line_style, label=metrics[metric])
        line_style_handles.append(line)

    # Show only every second epoch label
    for label in ax.xaxis.get_ticklabels()[::2]:
        label.set_visible(False)

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    ax.legend(handles=line_style_handles, loc='lower right')
    ax.add_artist(classifier_legend)

    fig.savefig(path.join(config.RESULT_DIR, selected_embeddings, fig_title))
    plt.show(block=fig_number == number_of_figures + 1)


def main():
    global data, number_of_figures
    data = ResultsData(config.METHODS, config.EMBEDDINGS, classifiers, num_epochs)
    number_of_figures = sum([len(data[method]) for method in data.methods])
    for method in data.methods:
        for embedding in data[method]:
            plot(method, embedding)


if __name__ == '__main__':
    main()
