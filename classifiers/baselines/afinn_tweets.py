from afinn import Afinn

from classifiers.baselines.method import Method


class AfinnTweets(Method):
    def __init__(self, tweets, tokenized_tweets, labels):
        Method.__init__(self, tweets, tokenized_tweets, labels, "AFINN")
        self.afinn = Afinn(emoticons=True)

    def classify(self, tweet, tokenized_tweet):

        score = self.afinn.score(tweet)

        if score > 0:
            return "positive"
        elif score < 0:
            return "negative"
        else:
            return "neutral"
