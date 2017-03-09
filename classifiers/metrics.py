from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

import methods


# Returns precision given predictions and labels
# Change current class with parameter version
# Converts neutrals to opposite of docs wanted (to -1 if current class is positives)
# predictions_num is vector with 1 for positive samples, 0 for neutral and -1 for negative
# labels_num is vector with 1 for positive samples, 0 for neutral and -1 for negative
def calc_precision_with_neutrals(predictions_num, labels_num, version="pos"):
    if version == "neg":
        rel_docs = -1
    else:
        rel_docs = 1

    # Change neutrals to opposite of docs wanted
    processed_true_vals = rel_docs * methods.remove_neutrals(labels_num, rel_docs)
    processed_pred_vals = rel_docs * methods.remove_neutrals(predictions_num, rel_docs)

    return precision_score(y_true=processed_true_vals, y_pred=processed_pred_vals)


# Returns recall given predictions and labels
# Change current class with parameter version
# Converts neutrals to opposite of docs wanted (to -1 if current class is positives)
# predictions_num is vector with 1 for positive samples, 0 for neutral and -1 for negative
# labels_num is vector with 1 for positive samples, 0 for neutral and -1 for negative
def calc_recall_with_neutrals(predictions_num, labels_num, version="pos"):
    if version == "neg":
        rel_docs = -1
    else:
        rel_docs = 1

    # Change neutrals to opposite of docs wanted
    processed_true_vals = rel_docs * methods.remove_neutrals(labels_num, rel_docs)
    processed_pred_vals = rel_docs * methods.remove_neutrals(predictions_num, rel_docs)

    return recall_score(y_true=processed_true_vals, y_pred=processed_pred_vals)


# Returns F1-score given predictions and labels_num
# Special version of F1-score, average of F-scores for positives and negatives
# predictions_num is vector with 1 for positive samples, 0 for neutral and -1 for negative
# labels_num is vector with 1 for positive samples, 0 for neutral and -1 for negative
def calc_F1_score_with_neutrals(predictions_num, labels_num):
    precision_pos = calc_precision_with_neutrals(predictions_num, labels_num, version="pos")
    recall_pos = calc_recall_with_neutrals(predictions_num, labels_num, version="pos")
    F1_pos = calc_F1_score(precision_pos, recall_pos)

    precision_neg = calc_precision_with_neutrals(predictions_num, labels_num, version="neg")
    recall_neg = calc_recall_with_neutrals(predictions_num, labels_num, version="neg")
    F1_neg = calc_F1_score(precision_neg, recall_neg)

    return (F1_pos + F1_neg) / 2


# Returns F1-score given precision and recall
def calc_F1_score(precision, recall):
    if precision + recall == 0:
        return 0

    return 2 * (precision * recall) / (precision + recall)
