import fjlc

from classifiers.models.model import Model


class LexiconClassifier(Model):
    def __init__(self):
        Model.__init__(self, "Lexicon Classifier")
        self.lexicon = fjlc.LexiconClassifier()

    def classify(self, tweet):
            sentiment = self.lexicon.classify(tweet)
            if sentiment == "POSITIVE":
                return 1
            elif sentiment == "NEGATIVE":
                return -1
            return 0

    def predict(self, tweets, embeddings_train_scaled):
        self.predictions = list(map(self.classify, tweets))
        return self.predictions
