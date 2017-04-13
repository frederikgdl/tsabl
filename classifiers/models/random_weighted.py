import numpy.random as numpy_random

from classifiers.models.model import Model


class RandomWeighted(Model):
    """
    Picks random label based on a probability distribution built from the test set
    """
    def __init__(self):
        Model.__init__(self, name="Random Weighted")
        self.probability_distribution = [1/3., 1/3., 1/3.]

    def train(self, tweets, embeddings, labels):
        num_tweets = float(len(tweets))
        p_positive = len([label for label in labels if label == 1]) / num_tweets
        p_negative = len([label for label in labels if label == -1]) / num_tweets
        p_neutral = len([label for label in labels if label == 0]) / num_tweets

        self.probability_distribution = [p_positive, p_negative, p_neutral]

    def predict(self, tweets, embeddings_train_scaled):
        self.predictions = [numpy_random.choice([1, -1, 0], p=self.probability_distribution) for t in tweets]
        return self.predictions
