from plots.results_data import ResultsData
from plots import config
from utils.misc import sorted_by_suffix

name_of_classifier = config.CLASSIFIER.lower()
metric = config.METRIC[0]
metric_pretty = config.METRIC[1]

num_epochs = config.NUM_EPOCHS

col_align = "c"
specificity = 4  # Number of decimals to round values to


def bold(text):
    return "\\textbf{" + str(text) + "}"


def create_tables(data):
    """
    Creates a table corresponding to combined_plot for each 'method' (binary, ternary, agg_ternary)
    PS! Works only for one classifier!
    :param data: ResultsData object 
    """
    for method in data.methods:

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


def main():
    data = ResultsData(config.METHODS, config.EMBEDDINGS, [name_of_classifier], num_epochs)
    create_tables(data)


if __name__ == "__main__":
    main()
