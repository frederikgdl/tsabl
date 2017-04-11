import classifiers.metrics as metrics


class Results:
    """
    A class for calculating and holding classifier evaluation metrics
    """

    def __init__(self, predictions, truth):
        self.predictions = predictions
        self.truth = truth

        self.results = {
            "f1_pn_score": metrics.f1_pn_score(predictions, truth),
            "ternary_macro_f1_score": metrics.ternary_macro_f1_score(predictions, truth),

            "positive_precision": metrics.ternary_precision(predictions, truth, 1),
            "positive_recall": metrics.ternary_precision(predictions, truth, 1),
            "positive_f1_score": metrics.ternary_f1_score(predictions, truth, 1),

            "negative_precision": metrics.ternary_precision(predictions, truth, -1),
            "negative_recall": metrics.ternary_precision(predictions, truth, -1),
            "negative_f1_score": metrics.ternary_f1_score(predictions, truth, -1),

            "neutral_precision": metrics.ternary_precision(predictions, truth, 0),
            "neutral_recall": metrics.ternary_precision(predictions, truth, 0),
            "neutral_f1_score": metrics.ternary_f1_score(predictions, truth, 0),
        }

    def __str__(self):
        """
        Return string representation of results
        :return: 
        """
        s = ""
        for key, value in self.results.items():
            s += '{:32}'.format(key) + '{:>10}'.format(str(value)) + "\n"

        return s
