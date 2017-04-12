from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

POSITIVE = 1
NEUTRAL = 0
NEGATIVE = -1


# Calculates the F1_PN-score for the given predictions and labels
# F1_PN is the average of F1-scores for positives and negatives
def f1_pn_score(predictions, labels):
    f1_pos = ternary_f1_score(predictions, labels, POSITIVE)
    f1_neg = ternary_f1_score(predictions, labels, NEGATIVE)
    return (f1_pos + f1_neg) / 2


def ternary_macro_f1_score(predictions, labels):
    f1_pos = ternary_f1_score(predictions, labels, POSITIVE)
    f1_neg = ternary_f1_score(predictions, labels, NEGATIVE)
    f1_neu = ternary_f1_score(predictions, labels, NEUTRAL)
    return (f1_pos + f1_neg + f1_neu) / 3


# Calculates F1-score for three-way classification
# target_value is the value of the class to calculate F1-score for
def ternary_f1_score(predictions, labels, target_value):
    precision = ternary_precision(predictions, labels, target_value)
    recall = ternary_recall(predictions, labels, target_value)
    return calc_f1_score(precision, recall)


# Calculates precision for three-way classification
def ternary_precision(predictions, labels, target_value):
    prepared_predictions = prepare_data(predictions, target_value)
    prepared_labels = prepare_data(labels, target_value)
    return precision_score(y_true=prepared_labels, y_pred=prepared_predictions)


# Calculates recall for three-way classification
def ternary_recall(predictions, labels, target_value):
    prepared_predictions = prepare_data(predictions, target_value)
    prepared_labels = prepare_data(labels, target_value)
    return recall_score(y_true=prepared_labels, y_pred=prepared_predictions)


# Prepares lists of labels or predictions for further scoring
# Sets the values with the target value to 1, the rest to -1
# This way we can consider only one target value at a time
def prepare_data(values, target_value):
    return [1 if value == target_value else -1 for value in values]


# Returns F1-score given precision and recall
def calc_f1_score(precision, recall):
    if precision + recall == 0:
        return 0

    return 2 * (precision * recall) / (precision + recall)
