from plots.results_data import ResultsData
from plots import config

name_of_classifier = "svm c=1"  # Must be lower case
metric = list(config.METRICS.keys())[1]  # Only one metric
metric_pretty = config.METRICS[metric]

num_epochs = config.NUM_EPOCHS

column_alignment = "c"
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

        num_columns = len(data[method]) + 1
        column_setup = "{" + "|".join([column_alignment] * num_columns) + "}"

        prefix = "\\begin{table}[H]\n\t\\centering\n\t\\begin{tabular}" + column_setup + "\n"

        # Create header
        header = bold("Epoch") + " & "
        for embedding in data[method]:
            header += bold(embedding) + " & "

        table = prefix + header[:-3] + " \\\\\n\\hline\n"

        for epoch in range(num_epochs):
            s = str(epoch + 1) + " & "
            for embedding in data[method]:
                value = round(data[method][embedding][name_of_classifier][epoch][metric], specificity)
                s += str(value) + " & "
            table += s[:-3] + " \\\\\n"

        postfix = "\t\\end{tabular}\n\\end{table}\n"
        table += postfix

        print(table)


def main():
    data = ResultsData(config.METHODS, config.EMBEDDINGS, [name_of_classifier], num_epochs)
    create_tables(data)


if __name__ == "__main__":
    main()
