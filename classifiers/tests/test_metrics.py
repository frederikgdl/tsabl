import unittest
import classifiers.metrics as metrics


class TestMetrics(unittest.TestCase):

    def setUp(self):
        self.test_set = [-1, -1, -1, 0, 0, 0, 1, 1, 1, 1]

        self.decent_predictions = [-1, -1, 0, 0, 0, 1, 1, 1, 1, 1]
        self.correct_distribution_wrong_order = [0, 0, 1, 1, 1, 1, 0, -1, -1, -1]
        self.all_negative = [-1] * len(self.test_set)
        self.incorrect_length = [0, 0]

    def test_ternary_precision(self):
        for label in [-1, 0, 1]:
            # If predictions are the same as test labels, precision should be 1.
            self.assertEquals(1.0, metrics.ternary_precision(self.test_set, self.test_set, label))

        # If predictions are all negative, precision for negative should be 0.3, 0 for positive and neutral
        self.assertEquals(0.3, metrics.ternary_precision(self.all_negative, self.test_set, -1))
        self.assertEquals(0, metrics.ternary_precision(self.all_negative, self.test_set, 0))
        self.assertEquals(0, metrics.ternary_precision(self.all_negative, self.test_set, 1))

        # If correct distribution, but all in incorrect order, should be 0 for all
        self.assertEquals(0, metrics.ternary_precision(self.correct_distribution_wrong_order, self.test_set, -1))
        self.assertEquals(0, metrics.ternary_precision(self.correct_distribution_wrong_order, self.test_set, 0))
        self.assertEquals(0, metrics.ternary_precision(self.correct_distribution_wrong_order, self.test_set, 1))

        # If length of arguments are not equal, expect ValueError
        self.assertRaises(ValueError, metrics.ternary_precision, self.incorrect_length, self.test_set, -1)
        self.assertRaises(ValueError, metrics.ternary_precision, self.incorrect_length, self.test_set, 0)
        self.assertRaises(ValueError, metrics.ternary_precision, self.incorrect_length, self.test_set, 1)

    def test_ternary_recall(self):
        for label in [-1, 0, 1]:
            # If predictions are the same as test labels, recall should be 1.
            self.assertEquals(1.0, metrics.ternary_recall(self.test_set, self.test_set, label))

        # If predictions are all negative, recall for negative should be 1, 0 for positive and neutral
        self.assertEquals(1, metrics.ternary_recall(self.all_negative, self.test_set, -1))
        self.assertEquals(0, metrics.ternary_recall(self.all_negative, self.test_set, 0))
        self.assertEquals(0, metrics.ternary_recall(self.all_negative, self.test_set, 1))

        # If correct distribution, but all in incorrect order, should be 0 for all
        self.assertEquals(0, metrics.ternary_recall(self.correct_distribution_wrong_order, self.test_set, -1))
        self.assertEquals(0, metrics.ternary_recall(self.correct_distribution_wrong_order, self.test_set, 0))
        self.assertEquals(0, metrics.ternary_recall(self.correct_distribution_wrong_order, self.test_set, 1))

        # If length of arguments are not equal, expect ValueError
        self.assertRaises(ValueError, metrics.ternary_recall, self.incorrect_length, self.test_set, -1)
        self.assertRaises(ValueError, metrics.ternary_recall, self.incorrect_length, self.test_set, 0)
        self.assertRaises(ValueError, metrics.ternary_recall, self.incorrect_length, self.test_set, 1)

    def test_f1_pn_score(self):
        # Expect ternary f1 score to be average of f1 scores for positive and negative
        f1_pos = metrics.ternary_f1_score(self.decent_predictions, self.test_set, 1)
        f1_neg = metrics.ternary_f1_score(self.decent_predictions, self.test_set, -1)
        self.assertEquals((f1_pos + f1_neg) / 2, metrics.f1_pn_score(self.decent_predictions, self.test_set))

    def ternary_macro_f1_score(self):
        # Expect ternary f1 score to be average of f1 scores for positive and negative
        f1_pos = metrics.ternary_f1_score(self.decent_predictions, self.test_set, 1)
        f1_neg = metrics.ternary_f1_score(self.decent_predictions, self.test_set, -1)
        f1_neu = metrics.ternary_f1_score(self.decent_predictions, self.test_set, 0)
        self.assertEquals((f1_pos + f1_neg + f1_neu) / 3, metrics.ternary_macro_f1_score(self.decent_predictions, self.test_set))

if __name__ == '__main__':
    unittest.main()
