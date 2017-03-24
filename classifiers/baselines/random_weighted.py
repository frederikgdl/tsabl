import numpy.random as numpy_random

from classifiers.baselines.method import Method


class RandomWeighted(Method):
    """
    Picks random label based on a probability distribution built from the test set
    """
    def __init__(self, tweets, tokenized_tweets, labels):
        Method.__init__(self, tweets, tokenized_tweets, labels, "RandomWeighted")

        num_tweets = float(len(tweets))
        p_positive = len([label for label in labels if label == 1]) / num_tweets
        p_negative = len([label for label in labels if label == -1]) / num_tweets
        p_neutral = len([label for label in labels if label == 0]) / num_tweets

        self.probability_distribution = [p_positive, p_negative, p_neutral]

    def classify(self, tweet, tokenized_tweet):
        return numpy_random.choice(["positive", "negative", "neutral"], p=self.probability_distribution)
