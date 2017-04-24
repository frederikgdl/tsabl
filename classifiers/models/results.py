import copy
from collections import OrderedDict
import classifiers.metrics as metrics


class Results:
    """
    A class for calculating and holding classifier evaluation metrics
    """

    def __init__(self, predictions, truth, results=None):
        self.predictions = predictions
        self.truth = truth

        if results is not None:
            self.results = results
            return

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

    def apply_operator(self, operand, operator):
        """
        Add metrics of other results instance to this instance's metrics

        :param operand: Operand
        :param operator: Operator function that takes self and operand as arguments
        :type operator: function
        :return: A clone of this Results instance with updated metrics
        """
        if not isinstance(operand, (Results, int, float)):
            return NotImplemented

        clone = self.clone()

        for key, value in clone.results.items():
            other_value = operand.results[key] if isinstance(operand, Results) else operand
            clone.results[key] = operator(clone.results[key], other_value)

        return clone

    def __add__(self, other):
        return self.apply_operator(other, lambda a, b: a + b)

    def __sub__(self, other):
        return self.apply_operator(other, lambda a, b: a - b)

    def __mul__(self, other):
        return self.apply_operator(other, lambda a, b: a * b)

    def __truediv__(self, other):
        return self.apply_operator(other, lambda a, b: a / float(b))

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def clone(self):
        return Results(copy.deepcopy(self.predictions), copy.deepcopy(self.truth), copy.deepcopy(self.results))
