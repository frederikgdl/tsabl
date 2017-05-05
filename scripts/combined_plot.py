from os import path, listdir, environ

import matplotlib
import matplotlib.patches as mpatches

# Fix for running this script on a server without graphics.
# This line must run before importing pyplot!
if 'DISPLAY' not in environ:
    matplotlib.use('Agg')

import matplotlib.pyplot as plt

from scripts import config

name_of_classifier = "SVM c=1"
metric = list(config.METRICS.keys())[0]  # Only one metric
metric_pretty = config.METRICS[metric]

num_epochs = config.NUM_EPOCHS

# { method: { dataset: [] } }
data = {}


# Plot
def plot(method):
    x = list(range(1, 1 + num_epochs))
    colors = ['r', 'g', 'b', 'k']
    line_style = '-'

    fig = plt.gcf()
    fig_title = method
    fig.canvas.set_window_title(fig_title)

    plot_lines = []

    for i, dataset in enumerate(sorted(data[method].keys())):
        # Each dataset gets a color
        color = colors[i % len(colors)]
        line = plt.plot(x,
                        list(data[method][dataset]),
                        line_style,
                        color=color)
        plot_lines.append(line)

    # plt.axis([0, len(embeddings_files), 0, 1])
    plt.xticks(x)
    plt.xlabel('Epoch')
    plt.ylabel('Score')

    # Classifier legend (colors)
    color_patches = []
    for i, dataset in enumerate(sorted(data[method].keys())):
        color = colors[i % len(colors)]
        patch = mpatches.Patch(color=color, label=dataset)
        color_patches.append(patch)

    plt.legend(handles=color_patches, loc='upper right')
    plt.savefig(path.join(config.RESULT_DIR, method, fig_title))
    plt.show()
    plt.clf()


def add_data_value(method, embedding, value):
    if method not in data:
        data[method] = {}
    if embedding not in data[method]:
        data[method][embedding] = []
    data[method][embedding].append(value)


def load_results(method, embedding, results_dir):
    # Load results of SVM classifier
    for epoch in [d for d in listdir(results_dir) if path.isdir(path.join(results_dir, d))]:
        file_path = path.join(results_dir, epoch, name_of_classifier.lower())
        with open(file_path) as f:
            for line in f:
                result_metric = line.split()[0]
                value = line.split()[1]
                if result_metric == metric:
                    add_data_value(method, embedding, value)


def main():
    for method in config.METHODS:
        for embedding in config.EMBEDDINGS:
            selected_embeddings = path.join(method, embedding)
            results_dir = path.join(config.RESULT_DIR, selected_embeddings)

            if not path.exists(results_dir):
                print("Skipping", method, embedding, "because its results_dir does not exist")
                continue

            if len([d for d in listdir(results_dir) if path.isdir(path.join(results_dir, d))]) < num_epochs:
                print("Skipping", method, embedding, "because its results do not contain enough epoch directories")
                continue

            if not any([f for f in listdir(results_dir) if f.endswith(".png")]):
                print("Skipping", method, embedding,
                      "because no png was found in " + results_dir + ", which means it has not been tested.")
                continue

            print("Doing", method, embedding)

            load_results(method, embedding, results_dir)

        if method in data:
            plot(method)


if __name__ == "__main__":
    main()
