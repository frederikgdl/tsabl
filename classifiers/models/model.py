from classifiers.funcs import load_model, save_model
import classifiers.metrics as metrics


class Model:
    def __init__(self, name="Model"):
        """
        :param name: Name of the baseline method
        """
        self.name = name
        self.predictions = []
        self.num_tweets = -1
        self.run_time = -1
        self.f1_pn = 0.0
        self.model = None

    def train(self, tweets, train_embeddings, numeric_test_labels):
        return NotImplemented

    def predict(self, tweets, embeddings_train_scaled):
        return NotImplemented

    def test(self, test_labels):
        """
        Calculates the F1-score
        :type test_labels: list
        :param test_labels: Numeric test labels
        :return: F1-score
        """
        if len(self.predictions) < len(test_labels):
            print("ERROR", self.name, "predictions list not same length as test_labels")
        self.f1_pn = metrics.f1_pn_score(self.predictions, test_labels)

        return self

    def load_model(self, model_file):
        self.model = load_model(model_file)

    def save_model(self, out_file):
        if not self.model:
            return
        save_model(self.model, out_file)

    def save_results(self, out_file):
        with open(out_file, "w+") as f:
            f.write(str(self.f1_pn) + "\n")

    def print(self):
        """
        Prints a line of execution stats
        :return: self
        """
        print('{:32}'.format("F1-score " + self.name + ":") + '{:>10}'.format(str(self.f1_pn)))
        return self
