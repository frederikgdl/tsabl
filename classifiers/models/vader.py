from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from classifiers.models.model import Model


class Vader(Model):
    def __init__(self, threshold=0.3):
        Model.__init__(self, "VADER")
        self.vader = SentimentIntensityAnalyzer()
        self.threshold = threshold

    def classify(self, tweet):
            vader_score = self.vader.polarity_scores(tweet)
            if vader_score['pos'] >= self.threshold:
                return 1
            elif vader_score['neg'] >= self.threshold:
                return -1
            return 0

    def predict(self, tweets, embeddings_train_scaled):
        self.predictions = list(map(self.classify, tweets))
        return self.predictions
