from classifiers.funcs import load_model, save_model
import classifiers.metrics as metrics
from classifiers.models.results import Results


class Model:
    def __init__(self, name="Model"):
        """
        :param name: Name of the baseline method
        """
        self.name = name
        self.predictions = []
        self.num_tweets = -1
        self.run_time = -1
        self.model = None
        self.results = None

    def train(self, tweets, train_embeddings, numeric_test_labels):
        return NotImplemented

    def predict(self, tweets, embeddings_train_scaled):
        return NotImplemented

    def test(self, test_labels):
        """
        Calculates the scores from predictions
        :type test_labels: list
        :param test_labels: Numeric test labels
        :return: F1-score
        """
        if len(self.predictions) < len(test_labels):
            print("ERROR", self.name, "predictions list not same length as test_labels")

        self.results = Results(self.predictions, test_labels)

        return self

    def load_model(self, model_file):
        self.model = load_model(model_file)

    def save_model(self, out_file):
        if not self.model:
            return
        save_model(self.model, out_file)

    def save_results(self, out_file):
        with open(out_file, "w+") as f:
            f.write(str(self.results))

    def print(self):
        """
        Prints a line of execution stats
        :return: self
        """
        print(self.name)
        print(self.results)
        return self
