from os import path, environ

import matplotlib
import matplotlib.patches as mpatches
from plots.results_data import ResultsData
from utils.latex import bold
from utils.misc import sorted_by_suffix


# Fix for running this script on a server without graphics.
# This line must run before importing pyplot!
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

col_align = "c"
specificity = 4  # Number of decimals to round values to


def print_table(method):
    """
    Creates a table corresponding to combined_plot for each 'method' (binary, ternary)
    PS! Works only for one classifier!
    :param method: Method   
    """
    print("% " + method + ": " + ", ".join(data[method].keys()))

    num_columns = len(data[method]) + 2
    column_setup = "{" + col_align + " " + col_align + "*{" + str(num_columns - 2) + "}{|" + col_align + "}}"

    prefix = "\\begin{table}[H]\n\t\\centering\n\t\\begin{tabular}" + column_setup + "\n"

    # Create header
    header = "\\multicolumn{" + str(num_columns) + "}{" + col_align + "}{" + bold(config.EMBEDDINGS_KEY) + "} \\\\\n"
    embeddings = sorted_by_suffix(data[method].keys())
    header_values = "& & " + " & ".join(list(map(lambda e: e.split("-")[-1], embeddings))) + " \\\\\n"

    hline = "\\hhline{~*{" + str(num_columns - 1) + "}{|-}}\n"

    epochs_header = "\\parbox[t]{2mm}{\\multirow{" + str(num_epochs) + "}{*}{\\rotatebox[origin=c]{90}{\\textbf{Epochs}}}}\n"

    table = prefix + header + header_values + hline + epochs_header

    max_value = -1

    for epoch in range(num_epochs):
        s = "& " + str(epoch + 1) + " & "
        for embedding in data[method]:
            value = round(data[method][embedding][name_of_classifier][epoch][metric], specificity)
            if value > max_value:
                max_value = value
            s += str(value) + " & "
        table += s[:-3] + " \\\\\n"

    caption = metric_pretty + " scores for " + method + " method on " + config.EMBEDDINGS_KEY + " embeddings."
    caption = caption.replace("_", "\\_")
    postfix = "\t\\end{tabular}\n\t\\caption{" + caption + "}\n\\end{table}\n"
    table += postfix

    table = table.replace(str(max_value), bold(str(max_value)))
    print(table)


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

    embeddings = sorted_by_suffix(data[method].keys())
    for i, embedding in enumerate(embeddings):
        # Each dataset gets a color
        color = colors[i % len(colors)]
        epoch_results = data[method][embedding][name_of_classifier][:num_epochs]
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
        print_table(method)

if __name__ == "__main__":
    main()
