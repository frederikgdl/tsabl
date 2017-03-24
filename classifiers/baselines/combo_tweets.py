from afinn import Afinn
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from classifiers.baselines.method import Method


class ComboTweets(Method):
    """
    Uses a combination of AFINN, VADER and TextBlob.
    The scores from the three methods are normalized to be between -1 and 1, then combined as a weighted average.
    The weights a, b, c can be set as arguments.
    """
    def __init__(self, tweets, tokenized_tweets, labels, a=1, b=1, c=1, d=1):
        Method.__init__(self, tweets, tokenized_tweets, labels, "Combo")
        self.afinn = Afinn(emoticons=True)
        self.vader = SentimentIntensityAnalyzer()

        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def classify(self, tweet, tokenized_tweet):

        if self.a + self.b + self.c == 0:
            return "neutral"

        blob = TextBlob(tweet)
        polarity = blob.sentiment.polarity

        afinn_score = self.afinn.score(tweet)
        vader_score = self.vader.polarity_scores(tweet)

        # Normalize scores to be between -1 and +1
        norm_afinn = afinn_score / (5 * len(tweet.split()))
        norm_vader = vader_score["compound"]
        norm_blob = polarity

        score = (self.a * norm_afinn + self.b * norm_vader + self.c * norm_blob) / (self.a + self.b + self.c)

        if score > self.d / 10.:
            return "positive"
        if score < -self.d / 10.:
            return "negative"

        return "neutral"
