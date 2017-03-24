from textblob import TextBlob

from classifiers.baselines.method import Method


class TextblobTweets(Method):
    def __init__(self, tweets, tokenized_tweets, labels, subjectivity_threshold=0.1, polarity_threshold=0.4):
        Method.__init__(self, tweets, tokenized_tweets, labels, "TEXTBLOB")
        self.subjectivity_threshold = subjectivity_threshold
        self.polarity_threshold = polarity_threshold

    def classify(self, tweet, tokenized_tweet):

        # Calculate sentiment scores
        blob = TextBlob(tweet)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity

        if subjectivity <= self.subjectivity_threshold and abs(polarity) < self.polarity_threshold:
            return "neutral"
        elif polarity >= self.polarity_threshold:
            return "positive"
        elif polarity <= -self.polarity_threshold:
            return "negative"

        return "neutral"
