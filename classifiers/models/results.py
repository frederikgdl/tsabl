from collections import OrderedDict
import classifiers.metrics as metrics


class Results:
    """
    A class for calculating and holding classifier evaluation metrics
    """

    def __init__(self, predictions, truth):
        self.predictions = predictions
        self.truth = truth

        self.results = OrderedDict()
        self.results["f1_pn_score"] = metrics.f1_pn_score(predictions, truth)
        self.results["ternary_macro_f1_score"] = metrics.ternary_macro_f1_score(predictions, truth)

        self.results["positive_precision"] = metrics.ternary_precision(predictions, truth, 1)
        self.results["positive_recall"] = metrics.ternary_precision(predictions, truth, 1)
        self.results["positive_f1_score"] = metrics.ternary_f1_score(predictions, truth, 1)

        self.results["negative_precision"] = metrics.ternary_precision(predictions, truth, -1)
        self.results["negative_recall"] = metrics.ternary_precision(predictions, truth, -1)
        self.results["negative_f1_score"] = metrics.ternary_f1_score(predictions, truth, -1)

        self.results["neutral_precision"] = metrics.ternary_precision(predictions, truth, 0)
        self.results["neutral_recall"] = metrics.ternary_precision(predictions, truth, 0)
        self.results["neutral_f1_score"] = metrics.ternary_f1_score(predictions, truth, 0)

    def __str__(self):
        """
        Return string representation of results
        :return: string including all result metrics
        """
        results_string = ""

        for key, value in self.results.items():
            results_string += '{:32}'.format(key) + '{:>10}'.format(str(value)) + "\n"

        return results_string
