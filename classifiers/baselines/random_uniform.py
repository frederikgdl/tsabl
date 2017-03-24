from random import choice

from classifiers.baselines.method import Method


class RandomUniform(Method):
    def __init__(self, tweets, tokenized_tweets, labels):
        Method.__init__(self, tweets, tokenized_tweets, labels, "RandomUniform")

    def classify(self, tweet, tokenized_tweet):
        return choice(["positive", "negative", "neutral"])
