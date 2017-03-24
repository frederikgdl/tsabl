from time import time, strftime, gmtime

import classifiers.metrics as metrics

class Method:
    def __init__(self, tweets, tokenized_tweets, labels, name="Method"):
        """
        :param tweets: Array of tweet text strings
        :param tokenized_tweets: Array of arrays of tokenized tweets
        :param labels: Array of correct numeric sentiment labels (-1 for negative, 0 for neutral, 1 for positive)
        :param name: Name of the baseline method
        """

        self.tweets = tweets
        self.tokenized_tweets = tokenized_tweets
        self.labels = labels
        self.name = name
        self.predictions = [None] * len(tweets)
        self.num_tweets = -1
        self.run_time = -1
        self.score = None

    def classify(self, tweet, tokenized_tweet):
        """
        Must be implemented by subclass
        :param tweet: Tweet to classify
        :param tokenized_tweet: Tokenized tweet to classify
        :return: A string that is either "positive", "negative" or "neutral"
        """
        return None

    def run(self):
        start_time = time()

        for index, tweet in enumerate(self.tweets):

            predicted_label = self.classify(tweet, self.tokenized_tweets[index])

            if predicted_label == "neutral":
                self.predictions[index] = 0
            elif predicted_label == "positive":
                self.predictions[index] = 1
            elif predicted_label == "negative":
                self.predictions[index] = -1

        self.num_tweets = len(self.predictions)
        self.run_time = time() - start_time

        return self

    def test(self):
        """
        Calculates the F1-score
        :return: F1-score
        """
        self.score = metrics.calc_F1_score_with_neutrals(self.predictions, self.labels)

        return self

    def print(self):
        """
        Prints a line of execution stats
        :return: self
        """
        print("F1-score", self.name + ":\t\t" + str(self.score))
        return self
