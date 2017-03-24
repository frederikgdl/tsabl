from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from classifiers.baselines.method import Method


class VaderTweets(Method):
    def __init__(self, tweets, tokenized_tweets, labels, threshold=0.3):
        Method.__init__(self, tweets, tokenized_tweets, labels, "VADER")
        self.vader = SentimentIntensityAnalyzer()
        self.threshold = threshold

    def classify(self, tweet, tokenized_tweet):

        vader_score = self.vader.polarity_scores(tweet)

        if vader_score['pos'] >= self.threshold:
            return "positive"
        elif vader_score['neg'] >= self.threshold:
            return "negative"
        elif vader_score['neu'] >= self.threshold:
            return "neutral"

        return "neutral"
