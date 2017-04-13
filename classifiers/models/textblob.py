from textblob import TextBlob

from classifiers.models.model import Model


class Textblob(Model):
    def __init__(self, subjectivity_threshold=0.1, polarity_threshold=0.4):
        Model.__init__(self, name="TEXTBLOB")
        self.subjectivity_threshold = subjectivity_threshold
        self.polarity_threshold = polarity_threshold

    def classify(self, tweet):
        # Calculate sentiment scores
        blob = TextBlob(tweet)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity

        if subjectivity <= self.subjectivity_threshold and abs(polarity) < self.polarity_threshold:
            return 0
        elif polarity >= self.polarity_threshold:
            return 1
        elif polarity <= -self.polarity_threshold:
            self.predictions.append(-1)
            return -1
        return 0

    def predict(self, tweets, embeddings_train_scaled):
        self.predictions = list(map(self.classify, tweets))
        return self.predictions
